// Xiaohongshu note — blog post (zh):
// _posts/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline.zh.md
// Captions lifted from the post's own zh alt texts; pinned-profile caveats kept.

const FIG = 'blog/assets/figures/minimax-h3-modular-pipeline';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#0f766e', // feature: pipeline palette from blog/_config.yml
  colorDeep: '#115e59',
  colorDarkest: '#134e4a',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '一间车间' },
        '两台引擎',
      ],
      sub: 'MiniMax-H3：文生视频+音频 与 参考驱动 两套 DiT\n共享同一组零件',
      chips: [
        { v: '93.9%', label: '时间花在去噪' },
        { v: '85.8s', label: '4×H200 warm 全程' },
        { v: '2+1', label: '两套 DiT · 一组共享件' },
      ],
    },
    {
      type: 'figure',
      title: '一间车间，两台引擎',
      sub: '前台共用，引擎按请求选路',
      imgAbs: `${ROOT}/${FIG}/fig1-architecture.svg`,
      tag: '图1',
      caption: '前台（tokenizer + Qwen3-VL 编码器）读懂请求；FL2VA / Ref2VA 两台 DiT 按请求路由；共享 video/audio VAE 是同一条成品线。启动时决定装哪台 —— 请求永远选不了没装的那台。',
    },
    {
      type: 'figure',
      title: '时间都花在哪（4×H200 实测）',
      sub: '124 帧 · 1344×768 · 50 个 sigma 点',
      imgAbs: `${ROOT}/${FIG}/fig3-e2e-measured.svg`,
      tag: '图2',
      caption: 'warm 全程 85.845s：去噪 80.616s 占 93.9%，VAE 解码只占 2.2%，CPU 编 MP4 才 1.3s。结论很直接 —— 优化的火力应该全部对准去噪。',
    },
    {
      type: 'stats',
      title: '启动前先算三笔账',
      sub: 'storage、host RAM、HBM 是三份独立预算',
      stats: [
        { v: '135GiB', desc: '每个分区的 checkpoint', sub: '合并服务 ≈ 270GiB 存储 —— 先看磁盘装不装得下' },
        { v: '149.9s', desc: '进程启动到 /health', sub: 'imports 51s + worker/NCCL 37s + 模型加载 60s' },
        { v: '50→49', desc: 'sigma 点 → 去噪次数', sub: 'num_inference_steps=N 是 N 个点、N−1 次评估' },
      ],
      footnote: '⚠️ 85.8s 是 pinned profiled 条件（FL2VA-only T2VA、U4/R1、BF16+CUDNN），不能外推到所有请求；unprofiled 对照波动大，profiler 开销未定量。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '含可复核的 4×H200 完整证据链：manifest、原始日志、输出哈希全公开',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（系列 Blog 1）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — H3 系列第一篇，还有 Turbo LoRA 加速篇' },
      ],
      disclaimer: '本文为 MiniMax-H3 优化系列 Blog 1（shipped 基线）；Turbo/量化/offload 各有后续篇。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '两套DiT共用一间车间',
    body: [
      'MiniMax-H3 有两套生成引擎：文生视频+音频（T2VA/FL2VA）一套，参考素材驱动（Ref2VA）一套。看起来像两台完整的机器，其实大部分零件是同一个：tokenizer、Qwen3-VL 编码器、video/audio VAE 全都一样。',
      '',
      'PR #5720 把它整理成“一间车间、两台专用引擎”🏭：',
      '🔹 共享组件只装一次，两套 DiT 围绕它们构建',
      '🔹 --task-type 是启动时的断路器：装哪台，决定能接哪些任务',
      '🔹 每个请求再用 task 参数选路 —— 请求选不了没装的那台',
      '🔹 不完整的 checkpoint 启动时直接失败（fail-closed），不带病上线',
      '',
      '4×H200 实测（pinned 口径）：',
      '⏱️ warm 全程 85.845±0.161s，其中去噪 80.616s = 93.9%',
      '🎬 VAE 解码只占 2.2%，CPU 编 MP4 才 1.3s',
      '💾 合并服务 ≈ 270GiB 存储；启动到就绪 ~150s',
      '',
      '最容易搞错的细节：num_inference_steps=50 是 50 个 sigma 点、49 次去噪 —— 数点数，不是数步😅',
      '',
      '📘 完整拆解（含完整证据链）：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'AIGC', '视频生成', 'vLLM', '推理优化'],
  },
});
