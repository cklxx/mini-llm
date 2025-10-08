# 3.3 分词策略与信息压缩

> 结合 `src/tokenizer/tokenizer_manager.py` 的实现，逐行说明 tokenizer 缓存与训练逻辑。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `hash_obj.update(line.encode('utf-8'))` | 计算数据内容的 MD5 指纹。 | 用数据分布决定是否需要重新训练 tokenizer。 |
| `hash_obj.update(str(stat.st_size).encode())`<br>`hash_obj.update(str(stat.st_mtime).encode())` | 将文件大小与修改时间纳入指纹。 | 避免仅内容采样导致的碰撞，确保缓存可靠。 |
| `cache_key = config.get_cache_key()` | 以配置生成唯一键。 | 同一数据在不同 vocab/类型下需要独立缓存。 |
| `if not force_retrain and self._is_cache_valid(...):` | 判断缓存是否有效。 | 在配置和数据未变时直接复用，节省训练时间。 |
| `tokenizer = BPETokenizer(vocab_size=config.vocab_size)` | 加载或新建 BPE tokenizer。 | 确保后续 encode/decode 行为与训练时一致。 |

## 实践建议
- 当你调整 `vocab_size` 时，缓存键会随之变化，无需手动清理旧模型。
- 若数据非常大，可增大 `_get_data_hash` 的 `max_lines`，提升指纹准确度。 
