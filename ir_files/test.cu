// main.cpp
//
// 用于加载并执行外部 PTX 文件中的 kernel "mma"，并展示结果和调试方法
//
// 使用方法：
//   1. 将 PTX 代码保存为 ptx_mma.ptx，或通过命令行传递路径。
//   2. 编译： nvcc main.cpp -lcuda -o run_ptx
//   3. 运行： ./run_ptx [ptx文件路径]
//   4. 调试： cuda-gdb ./run_ptx --args [ptx文件路径]

#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <cstdint>  // for intptr_t

#define CHECK_DRV(call)                                  \
  do {                                                   \
    CUresult err = call;                                 \
    if (err != CUDA_SUCCESS) {                           \
      const char* errStr;                                \
      cuGetErrorString(err, &errStr);                    \
      std::cerr << "CUDA 驱动 API 错误：" << errStr       \
                << " @ " << #call << std::endl;          \
      std::exit(EXIT_FAILURE);                           \
    }                                                    \
  } while (0)

int main(int argc, char** argv) {
  // 1. 初始化驱动 API 并创建上下文
  CHECK_DRV(cuInit(0));
  CUdevice device;
  CHECK_DRV(cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK_DRV(cuCtxCreate(&context, 0, device));

  // 2. 读取 PTX 文件（支持命令行参数指定路径）
  const char* ptxPath = (argc > 1) ? argv[1] : "ptx_mma.ptx";
  std::ifstream ptxFile(ptxPath);
  if (!ptxFile) {
    std::cerr << "无法打开 PTX 文件：" << ptxPath << std::endl;
    return -1;
  }
  std::string ptx((std::istreambuf_iterator<char>(ptxFile)),
                  std::istreambuf_iterator<char>());

  // 自动修正 .version 和 .target，以兼容本地 ptxas/驱动 JIT
  std::string patched = ptx;
  // 将 .version 改成 7.0
  if (auto p = patched.find(".version"); p != std::string::npos) {
    auto e = patched.find('\n', p);
    patched.replace(p, e - p, ".version 7.0");
  }
  // 将 .target 改成 sm_80
  if (auto p = patched.find(".target"); p != std::string::npos) {
    auto e = patched.find('\n', p);
    patched.replace(p, e - p, ".target sm_80");
  }

  // 3. JIT 加载
  CUmodule module;
  CUjit_option opts[2] = {
      CU_JIT_TARGET_FROM_CUCONTEXT,  // 根据 context 推断目标架构
      CU_JIT_FALLBACK_STRATEGY       // 不支持时回退到 PTX JIT
  };
  void* vals[2] = {
      nullptr,
      (void*)(intptr_t)CU_PREFER_PTX
  };
  CHECK_DRV(cuModuleLoadDataEx(&module,
                               patched.c_str(),
                               2, opts, vals));
  CUfunction kernel;
  CHECK_DRV(cuModuleGetFunction(&kernel, module, "mma"));

  // 4. 分配 Host 和 Device 内存
  const int M = 64, K = 64;
  size_t bytes = M * K * sizeof(float);
  
  // 初始化 Host 数据
  std::vector<float> h_A(M*K, 2.0f);  // 将 A 初始化为 1.0f
  std::vector<float> h_B(M*K, 3.14f); // 将 B 初始化为 3.14f
  std::vector<float> h_C(M*K, 0.0f);  // C 为 0
  std::vector<float> h_D(M*K, 0.0f);  // D 为 0，准备接收 kernel 计算结果

  CUdeviceptr d_A, d_B, d_C, d_D;
  CHECK_DRV(cuMemAlloc(&d_A, bytes));
  CHECK_DRV(cuMemAlloc(&d_B, bytes));
  CHECK_DRV(cuMemAlloc(&d_C, bytes));
  CHECK_DRV(cuMemAlloc(&d_D, bytes));

  // 将数据从 Host 传输到 Device
  CHECK_DRV(cuMemcpyHtoD(d_A, h_A.data(), bytes));
  CHECK_DRV(cuMemcpyHtoD(d_B, h_B.data(), bytes));
  CHECK_DRV(cuMemcpyHtoD(d_C, h_C.data(), bytes));
  CHECK_DRV(cuMemcpyHtoD(d_D, h_D.data(), bytes));

  // 5. 设置 kernel 参数并启动
  void* args[] = { &d_A, &d_B, &d_C, &d_D };  // 传递设备内存指针给 kernel
  unsigned gridX = 2, gridY = 2, gridZ = 1;
  unsigned blockX = 64, blockY = 2, blockZ = 1;
  CHECK_DRV(cuLaunchKernel(kernel,
                           gridX, gridY, gridZ,
                           blockX, blockY, blockZ,
                           (unsigned int)39000,
                           /*stream*/0,
                           args,
                           nullptr));

  // 6. 等待执行完毕并拷贝结果
  CHECK_DRV(cuCtxSynchronize());
  CHECK_DRV(cuMemcpyDtoH(h_D.data(), d_D, bytes));

  // 7. 打印前16个元素，以便调试
  std::cout << "输出前16个元素：\n";
  for (int i = 0; i < 16; ++i) {
    std::cout << h_D[i] << " ";
  }
  std::cout << std::endl;

  // 8. 清理
  cuMemFree(d_A);
  cuMemFree(d_B);
  cuMemFree(d_C);
  cuMemFree(d_D);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
