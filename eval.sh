#!/bin/bash

# CKPT_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero/ckpts/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000

# DREAMZERO_PATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero \
# bash examples/embodiment/dreamzero_libero_eval/run_server.sh \
#     --model-path "${CKPT_DIR}" \
#     --metadata-json-path "${CKPT_DIR}/experiment_cfg/metadata.json" \
#     --tokenizer-path /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero/ckpts/umt5-xxl \
#     --layer-skip 6,8,10,12,14,16,18,20,22,24,26,28,30,32,34 \
#     --device cuda:0 --port 8000


# 1. 记录开始时的系统时间（单位：秒）
start_time=$(date +%s)

for i in {0..9}; do
    echo "========================================"
    echo "🚀 开始运行 Task ID: $i"
    echo "========================================"
    
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LIBERO_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO \
    bash examples/embodiment/dreamzero_libero_eval/run_client.sh \
        --benchmark-name libero_spatial \
        --task-ids "$i" \
        --n-eval 50 \
        --save-video \
        --output-dir "./runs/libero_spatial_smoke/$i"

    echo "✅ Task ID: $i 执行完毕"
    echo ""
done

# 2. 记录结束时的系统时间
end_time=$(date +%s)

# 3. 计算耗时差值
cost_time=$((end_time - start_time))

echo "🎉 所有 10 个任务已全部执行完毕！"
echo "⏱️  总共耗时: $cost_time 秒"


MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
LIBERO_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO \
bash examples/embodiment/dreamzero_libero_eval/run_client.sh \
    --benchmark-name libero_spatial \
    --task-ids 1 \
    --n-eval 50 \
    --save-video \
    --output-dir "./runs/libero_spatial_smoke/