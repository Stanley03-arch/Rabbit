# Media — streaming converter demo

## Decode (after cloning)

```bash
cd docs/media
bash assemble_media.sh
# or manually:
# cat tiny_demo.mp4.b64.part* | base64 -d > tiny_demo.mp4
```

| File | Description |
|------|-------------|
| `tiny_demo.mp4` | Short (~4s) converter walkthrough (after decode) |
| `rabbit_streaming_converter.mp4` | Full ~18s demo if parts are present |
| `assemble_media.sh` | Rebuild MP4s from `*.b64.part*` |

The GitHub connector uploads text safely; binary MP4 is reconstructed with `base64 -d`.

Covers: ShardIndex → Mixtral/Qwen/DeepSeek scheme detect → pack routed + shared experts → `.rabbit`.
