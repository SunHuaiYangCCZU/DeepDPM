import re
import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO

# 配置参数
MAX_SEQ_LENGTH = 300  # 最大处理序列长度
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    """初始化模型（保持全精度）"""
    print("🔄 初始化模型中...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE).eval()  # 保持FP32精度
    print(f"✅ 模型已加载到 {DEVICE}（FP32模式）")
    return tokenizer, model


def process_sequence(tokenizer, model, raw_seq):
    """全精度处理单个序列"""
    try:
        # 预处理阶段
        clean_seq = raw_seq[:MAX_SEQ_LENGTH].upper()  # 强制截断
        clean_seq = re.sub(r"[UZOB]", "X", clean_seq)

        # 动态长度计算
        seq_length = min(len(clean_seq), MAX_SEQ_LENGTH)
        effective_max_length = seq_length + 2  # 包含特殊token

        # 生成输入（保持数据类型正确）
        inputs = tokenizer(
            " ".join(list(clean_seq)),
            add_special_tokens=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH + 2,
            truncation=True,
            return_tensors="pt"
        )

        # 确保数据类型正确
        input_ids = inputs["input_ids"].long().to(DEVICE)  # 必须保持Long类型
        attention_mask = inputs["attention_mask"].float().to(DEVICE)  # 保持FP32

        # 显存优化前检查
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            print(f"⏳ 当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # 模型推理
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 特征后处理
        embeddings = outputs.last_hidden_state[0][1:-1].cpu().numpy()

        # 动态填充
        if embeddings.shape[0] < MAX_SEQ_LENGTH:
            padding = np.zeros((MAX_SEQ_LENGTH - embeddings.shape[0], 1024))
            embeddings = np.vstack([embeddings, padding])

        return embeddings

    except Exception as e:
        print(f"❌ 处理错误: {str(e)}")
        return None


def main():
    """主处理流程"""
    tokenizer, model = initialize_model()

    # 读取数据
    input_path = "/home/ys/sunhuaiyang/predict/case_study/10.fasta"
    output_path = "/home/ys/sunhuaiyang/predict/case_study/feature/10T5_features.npy"

    print("\n📖 读取FASTA文件中...")
    sequences = [str(rec.seq) for rec in SeqIO.parse(input_path, "fasta")]
    print(f"✅ 成功加载 {len(sequences)} 条序列")

    # 处理控制
    features = []
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        print(f"\n🔍 处理进度 {idx + 1}/{len(sequences)} | 当前序列长度: {seq_len}")

        # 显存保护机制
        if seq_len > 300 and DEVICE.type == "cuda":
            print("⚠️ 检测到长序列，启用显存保护模式")
            torch.cuda.empty_cache()

        # 处理序列
        emb = process_sequence(tokenizer, model, seq)

        if emb is not None:
            # 维度验证
            assert emb.shape == (MAX_SEQ_LENGTH, 1024), f"维度异常: {emb.shape}"
            features.append(emb)
            print(f"🟢 特征生成成功 | 维度: {emb.shape}")

    # 保存结果
    np.save(output_path, np.array(features))
    print(f"\n💾 特征已保存至: {output_path}")
    print(f"最终维度: {np.array(features).shape}")

    exit()


if __name__ == "__main__":
    main()