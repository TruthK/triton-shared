#include <iostream>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda.h>

// PTX code (bare_matmul function) is assumed to be passed as ptxCode
const char* ptxCode = R"(
.version 8.4
.target sm_89
.address_size 64

        // .globl       bare_matmul             // -- Begin function bare_matmul
.extern .shared .align 16 .b8 __dynamic_shared_memory__[];
                                        // @bare_matmul
.visible .entry bare_matmul(
        .param .u64 .ptr .global .align 1 bare_matmul_param_0,
        .param .u64 .ptr .global .align 1 bare_matmul_param_1,
        .param .u64 .ptr .global .align 1 bare_matmul_param_2,
        .param .u32 bare_matmul_param_3,
        .param .u32 bare_matmul_param_4,
        .param .u64 .ptr .global .align 1 bare_matmul_param_5
)
.reqntid 64, 2, 1
{
        .reg .pred      %p<16>;
        .reg .b32       %r<98>;
        .reg .f32       %f<57>;
        .reg .b64       %rd<242>;

// %bb.0:
        ld.param.u64    %rd69, [bare_matmul_param_0];
        ld.param.u64    %rd70, [bare_matmul_param_1];
        mov.u32         %r33, %tid.x;
        cvt.s64.s32     %rd71, %r33;
        ld.param.u32    %r34, [bare_matmul_param_3];
        mov.u32         %r35, %ctaid.x;
        ld.param.u32    %r36, [bare_matmul_param_4];
        mov.u32         %r37, %ctaid.y;
        shl.b32         %r38, %r35, 5;
        shl.b32         %r39, %r37, 5;
        cvt.s64.s32     %rd72, %r39;
        cvt.s64.s32     %rd73, %r36;
        mul.wide.s32    %rd74, %r36, %r38;
        cvt.s64.s32     %rd1, %r34;
        mul.wide.s32    %rd75, %r34, %r38;
        add.s64         %rd2, %rd75, %rd72;
        mov.u32         %r40, %tid.y;
        cvt.s64.s32     %rd3, %r40;
        mul.wide.s32    %rd239, %r40, 32;
        setp.lt.s32     %p1, %r33, 0;
        shr.s32         %r41, %r33, 31;
        xor.b32         %r42, %r41, %r33;
        shr.s32         %r43, %r42, 31;
        shr.u32         %r44, %r43, 27;
        add.s32         %r45, %r42, %r44;
        shr.s32         %r46, %r45, 5;
        xor.b32         %r47, %r46, %r41;
        cvt.s64.s32     %rd5, %r47;
        mul.wide.s32    %rd6, %r47, 8;
        shr.u32         %r48, %r41, 30;
        add.s32         %r49, %r33, %r48;
        shr.s32         %r50, %r49, 2;
        cvt.s64.s32     %rd76, %r50;
        mul.wide.s32    %rd77, %r50, 4;
        setp.ne.s64     %p2, %rd77, %rd71;
        and.pred        %p3, %p1, %p2;
        selp.s64        %rd78, -1, 0, %p3;
        add.s64         %rd79, %rd78, %rd76;
        mul.wide.s32    %rd80, %r33, 4;
        shl.b64         %rd81, %rd79, 1;
        xor.b64         %rd82, %rd81, %rd80;
        and.b64         %rd83, %rd82, 12;
        shr.u32         %r51, %r41, 29;
        add.s32         %r52, %r33, %r51;
        shr.s32         %r53, %r52, 3;
        cvt.s64.s32     %rd84, %r53;
        mul.wide.s32    %rd85, %r53, 8;
        setp.ne.s64     %p4, %rd85, %rd71;
        and.pred        %p5, %p1, %p4;
        selp.s64        %rd86, -1, 0, %p5;
        add.s64         %rd7, %rd86, %rd84;
        and.b64         %rd8, %rd80, 28;
        xor.b64         %rd87, %rd7, %rd71;
        shl.b64         %rd88, %rd87, 2;
        and.b64         %rd9, %rd88, 28;
        shl.b64         %rd89, %rd79, 4;
        or.b64          %rd10, %rd89, %rd83;
        shl.b64         %rd90, %rd74, 2;
        add.s64         %rd91, %rd69, %rd90;
        mul.wide.s32    %rd92, %r39, 4;
        add.s64         %rd93, %rd91, %rd92;
        mul.lo.s64      %rd94, %rd79, %rd73;
        shl.b64         %rd95, %rd94, 2;
        add.s64         %rd96, %rd93, %rd95;
        shl.b64         %rd97, %rd2, 2;
        add.s64         %rd98, %rd70, %rd97;
        shl.b64         %rd99, %rd8, 2;
        add.s64         %rd11, %rd98, %rd99;
        setp.gt.s32     %p6, %r40, 0;
        mov.u32         %r54, %laneid;
        cvt.u64.u32     %rd12, %r54;
        and.b64         %rd13, %rd12, 15;
        shr.u64         %rd14, %rd12, 2;
        and.b64         %rd100, %rd14, 4;
        mul.wide.u32    %rd101, %r54, 2;
        and.b64         %rd102, %rd101, 12;
        xor.b64         %rd15, %rd100, %rd102;
        or.b64          %rd103, %rd100, 8;
        xor.b64         %rd16, %rd103, %rd102;
        and.b64         %rd17, %rd12, 3;
        or.b64          %rd18, %rd17, 4;
        shl.b64         %rd19, %rd17, 2;
        shl.b64         %rd20, %rd18, 2;
        or.b64          %rd21, %rd19, 16;
        bar.sync        0;
        shl.b64         %rd104, %rd10, 2;
        mov.u64         %rd105, __dynamic_shared_memory__;
        add.s64         %rd106, %rd105, %rd104;
        add.s64         %rd107, %rd106, 8192;
        shl.b64         %rd108, %rd80, 2;
        and.b64         %rd109, %rd108, 48;
        add.s64         %rd22, %rd96, %rd109;
        cp.async.cg.shared.global [%rd107], [%rd22], 16;
        shl.b64         %rd23, %rd7, 5;
        or.b64          %rd110, %rd23, %rd9;
        shl.b64         %rd111, %rd110, 2;
        add.s64         %rd112, %rd105, %rd111;
        mul.lo.s64      %rd24, %rd7, %rd1;
        shl.b64         %rd113, %rd24, 2;
        add.s64         %rd114, %rd11, %rd113;
        cp.async.cg.shared.global [%rd112], [%rd114], 16;
        cp.async.commit_group;
        cp.async.wait_group 0;
        bar.sync        0;
        @%p6 bra        $L__BB0_8;
// %bb.1:                               // %.lr.ph12
        cvt.u32.u64     %r55, %rd5;
        setp.lt.s32     %p7, %r55, 4;
        @%p7 bra        $L__BB0_4;
        bra.uni         $L__BB0_2;
$L__BB0_4:                              // %.lr.ph.us.preheader
        shl.b64         %rd115, %rd17, 7;
        add.s64         %rd25, %rd105, %rd115;
        shl.b64         %rd117, %rd18, 7;
        add.s64         %rd26, %rd105, %rd117;
        add.s64         %rd30, %rd6, -32;
        shl.b64         %rd129, %rd3, 12;
        mul.lo.s64      %rd130, %rd14, 96;
        or.b64          %rd131, %rd129, %rd130;
        shl.b64         %rd132, %rd5, 5;
        add.s64         %rd133, %rd131, %rd132;
        shl.b64         %rd134, %rd12, 3;
        add.s64         %rd135, %rd133, %rd134;
        add.s64         %rd137, %rd135, %rd105;
        add.s64         %rd228, %rd137, 16384;
        mov.u64         %rd229, %rd239;
$L__BB0_5:                              // %.lr.ph.us
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_6 Depth 2
        or.b64          %rd138, %rd13, %rd229;
        shl.b64         %rd139, %rd138, 4;
        or.b64          %rd140, %rd139, %rd15;
        shl.b64         %rd141, %rd140, 2;
        add.s64         %rd143, %rd105, 8192;
        add.s64         %rd144, %rd143, %rd141;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd144];
        or.b64          %rd145, %rd139, %rd16;
        shl.b64         %rd146, %rd145, 2;
        add.s64         %rd147, %rd143, %rd146;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r5, %r6, %r7, %r8}, [%rd147];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r9, %r10, %r11, %r12}, [%rd144+1024];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r13, %r14, %r15, %r16}, [%rd147+1024];
        mov.u64         %rd230, %rd228;
        mov.u64         %rd231, %rd30;
$L__BB0_6:                              //   Parent Loop BB0_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
        add.s64         %rd231, %rd231, 32;
        add.s64         %rd148, %rd231, %rd14;
        xor.b64         %rd149, %rd148, %rd19;
        shl.b64         %rd150, %rd149, 2;
        add.s64         %rd151, %rd25, %rd150;
        ld.shared.u32   %r72, [%rd151];
        xor.b64         %rd152, %rd148, %rd20;
        shl.b64         %rd153, %rd152, 2;
        add.s64         %rd154, %rd26, %rd153;
        ld.shared.u32   %r73, [%rd154];
        ld.shared.u32   %r74, [%rd151+1024];
        xor.b64         %rd155, %rd148, %rd21;
        shl.b64         %rd156, %rd155, 2;
        add.s64         %rd157, %rd25, %rd156;
        ld.shared.u32   %r75, [%rd157+1536];
        ld.shared.v2.f32        {%f1, %f2}, [%rd230];
        ld.shared.v2.f32        {%f3, %f4}, [%rd230+1024];
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f5, %f6, %f7, %f8},
                {%r1, %r2, %r3, %r4},
                {%r72, %r73},
                {%f1, %f2, %f3, %f4};
        ld.shared.v2.f32        {%f9, %f10}, [%rd230+2048];
        ld.shared.v2.f32        {%f11, %f12}, [%rd230+3072];
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f13, %f14, %f15, %f16},
                {%r9, %r10, %r11, %r12},
                {%r72, %r73},
                {%f9, %f10, %f11, %f12};
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f17, %f18, %f19, %f20},
                {%r5, %r6, %r7, %r8},
                {%r74, %r75},
                {%f5, %f6, %f7, %f8};
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f21, %f22, %f23, %f24},
                {%r13, %r14, %r15, %r16},
                {%r74, %r75},
                {%f13, %f14, %f15, %f16};
        st.shared.v2.f32        [%rd230], {%f17, %f18};
        st.shared.v2.f32        [%rd230+1024], {%f19, %f20};
        st.shared.v2.f32        [%rd230+2048], {%f21, %f22};
        st.shared.v2.f32        [%rd230+3072], {%f23, %f24};
        add.s64         %rd230, %rd230, 128;
        setp.lt.s64     %p9, %rd231, 0;
        @%p9 bra        $L__BB0_6;
// %bb.7:                               // %._crit_edge.us
                                        //   in Loop: Header=BB0_5 Depth=1
        add.s64         %rd38, %rd229, 32;
        add.s64         %rd228, %rd228, 4096;
        setp.lt.s64     %p10, %rd229, 0;
        mov.u64         %rd229, %rd38;
        @%p10 bra       $L__BB0_5;
        bra.uni         $L__BB0_8;
$L__BB0_2:                              // %.lr.ph12.split.preheader
        add.s64         %rd234, %rd239, -32;
        shl.b64         %rd118, %rd3, 9;
        or.b64          %rd119, %rd16, %rd118;
        shl.b64         %rd120, %rd13, 4;
        or.b64          %rd121, %rd119, %rd120;
        shl.b64         %rd122, %rd121, 2;
        add.s64         %rd124, %rd122, %rd105;
        add.s64         %rd233, %rd124, 9216;
        or.b64          %rd125, %rd15, %rd118;
        or.b64          %rd126, %rd125, %rd120;
        shl.b64         %rd127, %rd126, 2;
        add.s64         %rd128, %rd127, %rd105;
        add.s64         %rd232, %rd128, 9216;
$L__BB0_3:                              // %.lr.ph12.split
                                        // =>This Inner Loop Header: Depth=1
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r56, %r57, %r58, %r59}, [%rd232+-1024];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r60, %r61, %r62, %r63}, [%rd233+-1024];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r64, %r65, %r66, %r67}, [%rd232];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r68, %r69, %r70, %r71}, [%rd233];
        add.s64         %rd234, %rd234, 32;
        add.s64         %rd233, %rd233, 2048;
        add.s64         %rd232, %rd232, 2048;
        setp.lt.s64     %p8, %rd234, 0;
        @%p8 bra        $L__BB0_3;
$L__BB0_8:                              // %._crit_edge13
        ld.param.u64    %rd68, [bare_matmul_param_2];
        cvt.u32.u64     %r76, %rd3;
        setp.gt.s32     %p11, %r76, 0;
        bar.sync        0;
        add.s64         %rd161, %rd106, 10240;
        add.s64         %rd162, %rd22, 64;
        cp.async.cg.shared.global [%rd161], [%rd162], 16;
        add.s64         %rd163, %rd7, 16;
        shl.b64         %rd46, %rd163, 5;
        or.b64          %rd164, %rd46, %rd9;
        shl.b64         %rd165, %rd164, 2;
        add.s64         %rd166, %rd105, %rd165;
        mul.lo.s64      %rd47, %rd163, %rd1;
        shl.b64         %rd167, %rd47, 2;
        add.s64         %rd168, %rd11, %rd167;
        cp.async.cg.shared.global [%rd166], [%rd168], 16;
        cp.async.commit_group;
        cp.async.wait_group 0;
        bar.sync        0;
        @%p11 bra       $L__BB0_16;
// %bb.9:                               // %.lr.ph12.1
        cvt.u32.u64     %r77, %rd5;
        setp.lt.s32     %p12, %r77, 4;
        @%p12 bra       $L__BB0_12;
        bra.uni         $L__BB0_10;
$L__BB0_12:                             // %.lr.ph.us.1.preheader
        shl.b64         %rd169, %rd17, 7;
        add.s64         %rd48, %rd105, %rd169;
        add.s64         %rd52, %rd6, -32;
        shl.b64         %rd182, %rd3, 12;
        mul.lo.s64      %rd183, %rd14, 96;
        or.b64          %rd184, %rd182, %rd183;
        shl.b64         %rd185, %rd5, 5;
        add.s64         %rd186, %rd184, %rd185;
        shl.b64         %rd187, %rd12, 3;
        add.s64         %rd188, %rd186, %rd187;
        add.s64         %rd190, %rd188, %rd105;
        add.s64         %rd238, %rd190, 16384;
$L__BB0_13:                             // %.lr.ph.us.1
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_14 Depth 2
        or.b64          %rd191, %rd13, %rd239;
        shl.b64         %rd192, %rd191, 4;
        or.b64          %rd193, %rd192, %rd15;
        shl.b64         %rd194, %rd193, 2;
        add.s64         %rd196, %rd105, 8192;
        add.s64         %rd197, %rd196, %rd194;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r17, %r18, %r19, %r20}, [%rd197+2048];
        or.b64          %rd198, %rd192, %rd16;
        shl.b64         %rd199, %rd198, 2;
        add.s64         %rd200, %rd196, %rd199;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r21, %r22, %r23, %r24}, [%rd200+2048];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r25, %r26, %r27, %r28}, [%rd197+3072];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r29, %r30, %r31, %r32}, [%rd200+3072];
        mov.u64         %rd240, %rd238;
        mov.u64         %rd241, %rd52;
$L__BB0_14:                             //   Parent Loop BB0_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
        add.s64         %rd241, %rd241, 32;
        add.s64         %rd201, %rd241, %rd14;
        xor.b64         %rd202, %rd201, %rd19;
        shl.b64         %rd203, %rd202, 2;
        add.s64         %rd204, %rd48, %rd203;
        ld.shared.u32   %r94, [%rd204+2048];
        xor.b64         %rd205, %rd201, %rd20;
        shl.b64         %rd206, %rd205, 2;
        add.s64         %rd207, %rd48, %rd206;
        ld.shared.u32   %r95, [%rd207+2560];
        ld.shared.u32   %r96, [%rd204+3072];
        xor.b64         %rd208, %rd201, %rd21;
        shl.b64         %rd209, %rd208, 2;
        add.s64         %rd210, %rd48, %rd209;
        ld.shared.u32   %r97, [%rd210+3584];
        ld.shared.v2.f32        {%f25, %f26}, [%rd240];
        ld.shared.v2.f32        {%f27, %f28}, [%rd240+1024];
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f29, %f30, %f31, %f32},
                {%r17, %r18, %r19, %r20},
                {%r94, %r95},
                {%f25, %f26, %f27, %f28};
        ld.shared.v2.f32        {%f33, %f34}, [%rd240+2048];
        ld.shared.v2.f32        {%f35, %f36}, [%rd240+3072];
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f37, %f38, %f39, %f40},
                {%r25, %r26, %r27, %r28},
                {%r94, %r95},
                {%f33, %f34, %f35, %f36};
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f41, %f42, %f43, %f44},
                {%r21, %r22, %r23, %r24},
                {%r96, %r97},
                {%f29, %f30, %f31, %f32};
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {%f45, %f46, %f47, %f48},
                {%r29, %r30, %r31, %r32},
                {%r96, %r97},
                {%f37, %f38, %f39, %f40};
        st.shared.v2.f32        [%rd240], {%f41, %f42};
        st.shared.v2.f32        [%rd240+1024], {%f43, %f44};
        st.shared.v2.f32        [%rd240+2048], {%f45, %f46};
        st.shared.v2.f32        [%rd240+3072], {%f47, %f48};
        add.s64         %rd240, %rd240, 128;
        setp.lt.s64     %p14, %rd241, 0;
        @%p14 bra       $L__BB0_14;
// %bb.15:                              // %._crit_edge.us.1
                                        //   in Loop: Header=BB0_13 Depth=1
        add.s64         %rd66, %rd239, 32;
        add.s64         %rd238, %rd238, 4096;
        setp.lt.s64     %p15, %rd239, 0;
        mov.u64         %rd239, %rd66;
        @%p15 bra       $L__BB0_13;
        bra.uni         $L__BB0_16;
$L__BB0_10:                             // %.lr.ph12.split.1.preheader
        add.s64         %rd237, %rd239, -32;
        shl.b64         %rd171, %rd3, 9;
        or.b64          %rd172, %rd16, %rd171;
        shl.b64         %rd173, %rd13, 4;
        or.b64          %rd174, %rd172, %rd173;
        shl.b64         %rd175, %rd174, 2;
        add.s64         %rd177, %rd175, %rd105;
        add.s64         %rd236, %rd177, 11264;
        or.b64          %rd178, %rd15, %rd171;
        or.b64          %rd179, %rd178, %rd173;
        shl.b64         %rd180, %rd179, 2;
        add.s64         %rd181, %rd180, %rd105;
        add.s64         %rd235, %rd181, 11264;
$L__BB0_11:                             // %.lr.ph12.split.1
                                        // =>This Inner Loop Header: Depth=1
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r78, %r79, %r80, %r81}, [%rd235+-1024];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r82, %r83, %r84, %r85}, [%rd236+-1024];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r86, %r87, %r88, %r89}, [%rd235];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r90, %r91, %r92, %r93}, [%rd236];
        add.s64         %rd237, %rd237, 32;
        add.s64         %rd236, %rd236, 2048;
        add.s64         %rd235, %rd235, 2048;
        setp.lt.s64     %p13, %rd237, 0;
        @%p13 bra       $L__BB0_11;
$L__BB0_16:                             // %._crit_edge13.1
        or.b64          %rd211, %rd23, %rd8;
        shl.b64         %rd212, %rd211, 2;
        add.s64         %rd214, %rd105, 16384;
        add.s64         %rd215, %rd214, %rd212;
        ld.shared.v4.f32        {%f49, %f50, %f51, %f52}, [%rd215];
        add.s64         %rd217, %rd68, %rd97;
        add.s64         %rd219, %rd217, %rd113;
        add.s64         %rd221, %rd219, %rd99;
        st.global.f32   [%rd221+12], %f52;
        st.global.f32   [%rd221+8], %f51;
        st.global.f32   [%rd221+4], %f50;
        st.global.f32   [%rd221], %f49;
        or.b64          %rd222, %rd46, %rd8;
        shl.b64         %rd223, %rd222, 2;
        add.s64         %rd224, %rd214, %rd223;
        ld.shared.v4.f32        {%f53, %f54, %f55, %f56}, [%rd224];
        add.s64         %rd226, %rd217, %rd167;
        add.s64         %rd227, %rd226, %rd99;
        st.global.f32   [%rd227+12], %f56;
        st.global.f32   [%rd227+8], %f55;
        st.global.f32   [%rd227+4], %f54;
        st.global.f32   [%rd227], %f53;
        ret;
                                        // -- End function
}
)";

int main() {
    // Initialize CUDA Driver API
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    // Initialize CUDA driver
    cuInit(0);

    // Get CUDA device
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create CUDA context
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Load PTX code
    CUresult res = cuModuleLoadData(&cuModule, ptxCode);
    if (res != CUDA_SUCCESS) {
        std::cerr << "Error loading PTX code!" << std::endl;
        return -1;
    }

    // Get the function handle for the kernel
    res = cuModuleGetFunction(&cuFunction, cuModule, "bare_matmul");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Error getting function handle!" << std::endl;
        return -1;
    }

    // Matrix dimensions (512x512)
     int N = 512;
     size_t size = N * N * sizeof(float);

    // Allocate memory on host
    float *h_matrixA = (float*)malloc(size);
    float *h_matrixB = (float*)malloc(size);
    float *h_matrixC = (float*)malloc(size);

    // Initialize matrices with some values (example initialization)
    for (int i = 0; i < N * N; i++) {
        h_matrixA[i] = 2.0f;
        h_matrixB[i] = 2.0f;
    }

    // Allocate memory on device
    float *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc((void**)&d_matrixA, size);
    cudaMalloc((void**)&d_matrixB, size);
    cudaMalloc((void**)&d_matrixC, size);

    // Copy data to device
    cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);

    // Kernel parameters: matrixA, matrixB, matrixC, N
    void* kernelParams[] = { &d_matrixA, &d_matrixB, &d_matrixC, &N, &N, &d_matrixA };

    // Grid and block configuration (adjust as needed)
    dim3 threadsPerBlock(2,2,1);  // 16x16 threads per block
    dim3 numBlocks(16,16);        // 32x32 blocks

    // Launch the kernel
    res = cuLaunchKernel(cuFunction,
                         numBlocks.x, numBlocks.y, 1,   // grid size
                         threadsPerBlock.x, threadsPerBlock.y, 1,  // block size
                         0, NULL, kernelParams, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel!" << std::endl;
        return -1;
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_matrixC, d_matrixC, size, cudaMemcpyDeviceToHost);

    // Print part of the result (e.g., first few elements)
    for (int i = 0; i < 10; i++) {
        std::cout << h_matrixC[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixC);

    // Unload PTX module and destroy the context
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

    std::cout << "Kernel executed successfully!" << std::endl;
    return 0;
}
