from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes,llvm,tts_nv
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import tempfile
import os
import re
import signal
import subprocess
import functools
from pathlib import Path

currentDirname = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(currentDirname)
# 拼接目标路径,得到triton的python文件夹中backend的nvidia文件夹
dirname = os.path.join(parent_dir, "triton", "python", "triton", "backends","nvidia")

def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (16, 32, 16) if lhsType.is_int8() else (16, 16, 16)


@functools.lru_cache()
def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(dirname, "bin", binary),
    ]

    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return bin, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas_version():
    version = subprocess.check_output([_path_to_binary("ptxas")[0], "--version"]).decode("utf-8")
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
        elif minor == 6:
            return 85
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


@functools.lru_cache()
def get_features(options):
    ptx_version = options.ptx_version
    if ptx_version is None:
        _, cuda_version = _path_to_binary("ptxas")
        ptx_version = ptx_get_version(cuda_version)

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
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'

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
        return target.backend == 'kzx_cuda'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in KzxCUDAOptions.__dataclass_fields__.keys() if k in opts}
        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(KzxCUDAOptions.supported_fp8_dtypes)
            if self.capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dtypes" not in args:
            if self.capability >= 90:
                args["deprecated_fp8_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return KzxCUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        import triton.language.extra.cuda as cuda
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if self.capability >= 80 else cuda.convert_custom_float8_sm70,
            "min_dot_size": min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        tts_nv.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod
    
    @staticmethod
    def make_ttsharedir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        tts_nv.passes.tts.triton_to_linalg(pm)
        pm.run(mod)
        return mod


    @staticmethod
    def make_llir(mod,metadata, options, capability):
        # Get tts-MLIR as string
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        
        tts_nv.passes.convert.linalg_to_llvm(pm)
        
        pm.run(mod)
        
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()

        llvm_mod = llvm.to_module(mod, context)
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        features = get_features(options)
        triple = 'nvptx64-nvidia-cuda'
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        tts_nv.set_nvvm_reflect_ftz(llvm_mod)
        
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
    
    @staticmethod
    def make_ptx(src, metadata, opt, capability):
        ptx_version = opt.ptx_version
        if ptx_version is None:
            _, cuda_version = _path_to_binary("ptxas")
            ptx_version = ptx_get_version(cuda_version)

        triple = 'nvptx64-nvidia-cuda'
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        features = get_features(opt)
        ret = llvm.translate_to_asm(src, triple, proc, features, ['nvptx-short-ptr'], opt.enable_fp_fusion, False)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        # Remove the debug flag that prevents ptxas from optimizing the code
        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    @staticmethod
    def make_cubin(src, metadata, opt, capability):
        ptxas, _ = _path_to_binary("ptxas")
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            line_info = [] if os.environ.get('TRITON_DISABLE_LINE_INFO') else ['-lineinfo']
            fmad = [] if opt.enable_fp_fusion else ['--fmad=false']
            suffix = 'a' if capability == 90 else ''
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

                raise RuntimeError(f'{error}\n'
                                   f'`ptxas` stderr:\n{log}\n'
                                   f'Repro command: {ptxas_cmd}\n')

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin


    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: self.make_ttsharedir(src, metadata, options,self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src,metadata, options,self.capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)


    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version()
        return f'{version}-{self.capability}'
