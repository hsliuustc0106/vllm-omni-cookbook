// Xiaohongshu note — blog post:
// _posts/2026-08-18-online-quantization-fp8.md
// Memory-first framing; speed and per-model quality caveats kept in.

const FIG = 'blog/assets/figures/online-quantization-fp8';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · 特性深潜',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#7c3aed', // feature: quantization palette from blog/_config.yml
  colorDeep: '#5b21b6',
  colorDarkest: '#3b0764',

  cards: [
    {
      type: 'cover',
      lines: [
        '下载好的BF16模型',
        { text: '一个flag变成FP8' },
      ],
      sub: '加载时现场量化：无需校准、无需第二个 checkpoint\n权重字节直接 <b>省一半</b>',
      chips: [
        { v: '−50%', label: '量化后权重字节' },
        { v: 'W8A8', label: 'FP8 权重+激活' },
        { v: '0次', label: '校准 / 预处理' },
      ],
    },
    {
      type: 'figure',
      title: '三个阶段：配置 → 逐层加载 → 稳态服务',
      sub: '量化发生在加载的路上，不占额外一步',
      imgAbs: `${ROOT}/${FIG}/fig1.png`,
      tag: '图1',
      caption: '权重从 meta 设备逐层流式进卡：每层先以 BF16 物化、立刻量化成 FP8 并释放原存储。加载峰值 ≈ 累计 FP8 + 一层 BF16 —— 永远不会把整个 BF16 模型再装一遍。',
    },
    {
      type: 'figure',
      title: '两把尺子，两种命',
      sub: '为什么它敢说“免校准”',
      imgAbs: `${ROOT}/${FIG}/fig2.png`,
      tag: '图2',
      caption: '权重 scale 加载时算一次就冻结（max|W|/448）；激活 scale 每次前向现算 —— “dynamic”的全部含义。权重静态、激活动态；遇到敏感层（Qwen-Image 的 img_mlp）就用 ignored_layers 让它留在 BF16。',
    },
    {
      type: 'figure',
      title: '显存到底省在哪',
      sub: '磁盘不变，显存减半',
      imgAbs: `${ROOT}/${FIG}/fig3.png`,
      tag: '图3',
      caption: '磁盘上仍是原 BF16 checkpoint（每次启动重新量化一遍，这是代价）；显存里稳态只剩 FP8 权重 + 冻结 scale。保证省的是每元素 2字节→1字节，激活、attention 工作区、VAE 一概不动。',
    },
    {
      type: 'stats',
      title: '实测：fit 与不 fit 的区别',
      sub: '数字来自上游官方 recipe',
      stats: [
        { v: 'fit', desc: 'LTX-2.5 上 80G H100', sub: '在线 FP8 · 960×544 蒸馏单阶段；BF16 双阶段 1920×1088 ≈ 114GB，装不下' },
        { v: '−13.4%', desc: 'Qwen-Image 峰值显存', sub: '99.0 → 85.8 GB（2×B200，ModelOpt FP8 口径）' },
      ],
      footnote: '⚠️ 为什么只降 13%？权重减半只是峰值的一部分，激活/VAE/attention 工作区原封不动。显存是主要收益 —— Blackwell 上速度不一定快，可选 quack 包做融合后才快。',
    },
    {
      type: 'code',
      title: '上手：真的只有一个 flag',
      steps: [
        {
          chip: '①',
          label: 'BF16 checkpoint 直接服务',
          code: `vllm serve Lightricks/LTX-2.5-Diffusers \\
    --omni \\
    --model-class-name LTX2DistilledOneStagePipeline \\
    --quantization fp8`,
          note: 'Python 侧 Omni(model=..., quantization="fp8") 同效；ignored_layers 可以点名敏感层留在 BF16。',
        },
      ],
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '还有“逐文件”的代码路径走读：配置工厂、逐层加载器、HSDP 适配都讲了',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇文章' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — 在线量化篇， validated 模型清单和 ignored_layers 指南都在' },
      ],
      disclaimer: '显存数字来自上游 recipe（Qwen 行为离线 FP8 checkpoint 口径）。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '一个flag，显存直接省一半',
    body: [
      '想给视频生成模型省显存，以前的路径都很重：跑离线量化工具、发布第二个几十 GB 的 checkpoint、还得保持两边同步😫',
      '',
      'vLLM-Omni 的“在线量化”把这一步删了：加载 BF16 checkpoint 的现场，把每层顺手量化成 FP8（W8A8）——',
      '🔹 不用校准：激活 scale 每次前向现算（dynamic）',
      '🔹 不用第二个文件：量化权重只存在于显存',
      '🔹 权重字节直接砍半：2 字节 → 1 字节',
      '',
      '最直观的例子：LTX-2.5 BF16 双阶段 1920×1088 要 ~114GB，80G 的 H100 根本装不下；--quantization fp8 一个 flag，单卡就 fit 了✅',
      '',
      'Qwen-Image 实测峰值显存 99.0 → 85.8 GB（−13.4%，2×B200）。为什么只降 13%？权重减半只是峰值的一部分，激活/VAE/attention 工作区都不变。',
      '',
      '两个诚实的提醒⚠️：',
      '1️⃣ 省的是显存，不一定是速度 —— Blackwell 上 bias 是独立 kernel，装可选的 quack 包融合后才快',
      '2️⃣ 质量按模型验证 —— 敏感层 ignored_layers 留 BF16，先和 BF16 同 seed 对比再上线',
      '',
      '📘 完整拆解：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'AIGC', '深度学习', 'GPU', '模型量化'],
  },
});
