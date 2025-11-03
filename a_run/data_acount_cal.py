import tarfile
import pyarrow.parquet as pq
from io import BytesIO

def count_samples_in_parquet_tar(tar_path):
    """统计 .tar 包中所有 parquet 文件的样本总数"""
    total_rows = 0
    file_count = 0

    # 打开 tar 包
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            # 只统计 parquet 文件
            if member.isfile() and member.name.endswith(".parquet"):
                file_count += 1
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                f.close()
                # 在内存中解析 parquet 文件
                pf = pq.ParquetFile(BytesIO(data))
                total_rows += pf.metadata.num_rows

    print(f"📦 Tar 文件: {tar_path}")
    print(f"📂 Parquet 文件数量: {file_count}")
    print(f"📊 样本总数: {total_rows}")
    return total_rows


# === 示例调用 ===
if __name__ == "__main__":
    tar_path = "/primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.tar"
    count_samples_in_parquet_tar(tar_path)