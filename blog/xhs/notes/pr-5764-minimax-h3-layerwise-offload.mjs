// Xiaohongshu note — blog post (zh):
// _posts/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload.zh.md
// Caption lifted from the post's zh figure alt text; capacity-not-speed caveats kept.

const FIG = 'blog/assets/figures/pr-5764-minimax-h3-layerwise-offload';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#0891b2', // feature: offloader palette from blog/_config.yml
  colorDeep: '#0e7490',
  colorDarkest: '#155e75',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '近120G模型' },
        '塞进两张显卡',
      ],
      sub: 'MiniMax-H3 × 2× RTX 5090：DLO 逐层搬运\n不是更快，是装得下',
      chips: [
        { v: '118GB+', label: '模型权重总量' },
        { v: '64GB', label: '两卡 HBM 合计' },
        { v: '~22.6GiB', label: '实测单卡峰值' },
      ],
    },
    {
      type: 'figure',
      title: '权重住内存，GPU 逐层取',
      sub: 'PR #5764 · distributed layerwise offload',
      imgAbs: `${ROOT}/${FIG}/fig1-dlo-layout.svg`,
      tag: '图1',
      caption: '66.3GB 的 DiT 加 51.5GB 的 encoder 放在 host 的 pinned 内存里；每张 GPU 只常驻前 20 个 block，另留两个轮换槽位：算第 k 层时预取第 k+1 层。encoder 与 VAE 按需上卡，两个 GPU 各自只搬运自己的 TP 分片。',
    },
    {
      type: 'stats',
      title: '小操作台 + 门口冷库',
      sub: '不把所有食材摆上台面，用到哪样拿哪样',
      stats: [
        { v: '8分38秒', desc: '1344×768 · 50步', sub: '2× RTX 5090 完整跑通一个 5 秒视频（124帧）' },
        { v: '~22.6GiB', desc: '单卡采样峰值', sub: '32GB 的卡只用七成；ffmpeg 完整解码校验通过' },
        { v: '20 / 12', desc: '常驻层数起点', sub: '2×32GB 用 20；24GB 或单卡用 12（B300 容量 proxy）' },
      ],
      footnote: '⚠️ 这是容量特性不是速度特性：单次未预热运行，显存为 nvidia-smi 采样峰值；4090 档位只在 B300 上做过容量验证。host 内存才是大头：最低 200GB 内存 + 135GB 磁盘。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '含同周两个正确性修复（#5802 不等大 block、#5864 DP 并发）与怎么选的决策卡',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（系列 Blog 4）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — H3 系列第 4 篇，offload 主题核心篇' },
      ],
      disclaimer: '本文为 MiniMax-H3 优化系列 Blog 4（offload 主题核心篇）。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '近120G模型塞进2张显卡',
    body: [
      'MiniMax-H3 的权重有多大？DiT 66.3GB + 文本 encoder 51.5GB + 视频/音频 VAE——加起来近 120GB。而两张 RTX 5090 的显存合计只有 64GB，怎么跑？',
      '',
      'PR #5764 的答案：权重不住显卡，住内存（host pinned memory），GPU 一次只取一层🧅',
      '',
      '像小操作台 + 门口冷库：不必把所有食材摆上台面，用到哪样拿哪样；永远常用的几样（前 20 个 block）留在手边，两个轮换槽位算着第 k 层、预取第 k+1 层。',
      '',
      '2× RTX 5090 实测（1344×768、50 步、5 秒视频）：',
      '⏱️ 8 分 38 秒完整跑通，124 帧输出',
      '📊 单卡采样峰值 ~22.6GiB（32GB 卡用了七成）',
      '💾 代价在主机：最低 200GB 内存、135GB 磁盘/分区',
      '',
      '诚实说明：这是容量特性，不是加速——单次未预热运行，显存是 nvidia-smi 采样值；24GB（4090）档位只在 B300 上做过容量验证；调常驻层数省的是显存，host 内存不降。',
      '',
      '同一周还修了两个真问题：不等大的 transformer block 会让 AllGather 直接崩（#5802）；DP 并发请求的 wave 会挂死（#5864，加了派发前预检）。',
      '',
      '📘 完整拆解：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'vLLM', '推理优化', '视频生成'],
  },
});
