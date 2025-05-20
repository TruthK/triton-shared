from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes,llvm,ttsnv
from triton.runtime.errors import PTXASError
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import tempfile
import os
import re
import signal
import functools
import subprocess
from pathlib import Path
import sysconfig

currentDirname = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(currentDirname)
# 拼接目标路径,得到triton的python文件夹中backend的nvidia文件夹
dirname = os.path.join(parent_dir, "triton", "python", "triton", "backends","nvidia")

def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        if lhs_bitwidth == 8:
            return (16, 16, 32)
        else:
            return (16, 16, 16)

    return check_dot_compatibility

@functools.lru_cache()
def _path_to_binary(binary: str):
    binary += sysconfig.get_config_var("EXE")
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(dirname, "bin", binary),
    ]

    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return path, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas(arch: int):
    name = "ptxas-blackwell" if arch >= 100 else "ptxas"
    return _path_to_binary(name)


@functools.lru_cache()
def get_ptxas_version(arch: int):
    mock_ver = os.environ.get('TRITON_MOCK_PTX_VERSION')
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch)[0], "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        _, cuda_version = get_ptxas(arch)
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.3 is the max version supported by llvm 3a83162168.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(83, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class KzxCUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None

    def __post_init__(self):
        default_libdir = os.path.join(dirname, "lib")
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH", os.path.join(default_libdir,'libdevice.10.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KzxCUDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {'arch': os.getenv("TRITON_OVERRIDE_ARCH", f"sm{self.target.arch}")}
        args.update({k: opts[k] for k in KzxCUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(KzxCUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return KzxCUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.block_dims[0],
            metadata.block_dims[1],
            metadata.block_dims[2],
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        ttsnv.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        print(str(mod))
        return mod
    

    def make_llir(self, mod,metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)
        # Get tts-MLIR as string
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        ttsnv.passes.tts.triton_to_linalg(pm)
        ttsnv.passes.tts_codegen.iree_materialize_target(pm,capability,ptx_version)
        ttsnv.passes.tts_codegen.iree_llvmgpu_select_lowering_strategy(pm)
        ttsnv.passes.tts_codegen.iree_llvmgpu_codegen(pm,capability,ptx_version)
        ttsnv.passes.tts_codegen.llvmgpu_ptr_transform(pm)
        pm.run(mod)
        
        metadata["shared"] = mod.get_int_attr("ttg.shared")
        metadata["block_dims"] = mod.get_array_i32_attr("nvvm.reqntid")

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supporteedd on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        # use sm_90a until sm_100 is open sourced in llvm.
        if capability == 100:
            proc = 'sm_90a'
        features = get_features(options, self.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        ttsnv.set_nvvm_reflect_ftz(llvm_mod)

        # Set maxnreg on all kernels, if it was provided.
        if options.maxnreg is not None:
            for k in llvm_mod.get_functions():
                if not k.is_declaration() and k.is_external_linkage():
                    k.set_nvvm_maxnreg(options.maxnreg)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret
    
    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = 'nvptx64-nvidia-cuda'
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        # use sm_90a until sm_100 is open sourced in llvm.
        if capability == 100:
            proc = 'sm_90a'
        features = get_features(opt, self.target.arch)
        ret = llvm.translate_to_asm(src, triple, proc, features, [], opt.enable_fp_fusion, False)
        
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # Path(".vscode/tts_ptx_"+str(names[0])+".ir").write_text(str(ret))
        
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        # Remove the debug flag that prevents ptxas from optimizing the code
        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        # if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
        print("// -----// NVPTX Dump //----- //")
        print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas, _ = get_ptxas(self.target.arch)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            line_info = ["-lineinfo", "-suppress-debug-info"] if os.environ.get("TRITON_DISABLE_LINE_INFO",
                                                                                "0") == "1" else ["-lineinfo"]
            fmad = [] if opt.enable_fp_fusion else ['--fmad=false']
            suffix = 'a' if capability >= 90 else ''
            opt_level = ['--opt-level', '0'] if os.environ.get("DISABLE_PTXAS_OPT", "0") == "1" else []
            ptxas_cmd = [
                ptxas, *line_info, *fmad, '-v', *opt_level, f'--gpu-name=sm_{capability}{suffix}', fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                raise PTXASError(f"{error}\n"
                                 f"`ptxas` stderr:\n{log}\n"
                                 f'Repro command: {" ".join(ptxas_cmd)}\n')

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin


    def add_stages(self, stages, options):
        capability = self._parse_arch(options.arch)
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src,metadata, options,capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)


    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version(self.target.arch)
        return f'{version}-{self.target.arch}'
