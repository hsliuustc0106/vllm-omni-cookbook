// Xiaohongshu note — blog post (zh):
// _posts/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding.zh.md
// Caption lifted from the post's zh figure alt text; gap-level honesty caveats kept.

const FIG = 'blog/assets/figures/pr-6542-mask-free-packed-padding';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#b45309', // feature: host_path palette from blog/_config.yml
  colorDeep: '#92400e',
  colorDarkest: '#78350f',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '每次注意力前' },
        '白等半毫秒',
      ],
      sub: 'MiniMax-H3 × TRTLLM：GPU 明明算完了\n却在逐层重数 padding 的箱子',
      chips: [
        { v: '499.7→0.35', label: 'µs · 间隔 p50' },
        { v: '0', label: '中间 GPU 活动' },
        { v: '0', label: 'D2H 拷贝' },
      ],
    },
    {
      type: 'figure',
      title: '一次注意力调用：改前 vs 改后',
      sub: 'PR #6542 · mask-free packed padding',
      imgAbs: `${ROOT}/${FIG}/fig1-attention-gap.svg`,
      tag: '图1',
      caption: '之前：每次调用都要在 all-to-all 与 FMHA 之间做 mask 求和、前缀比较、nonzero 搜索和 10 次 D2H 同步——p50 间隔 499.7 µs。之后：后端直接读生产方在主机侧发布的 PackedPaddingMetadata 整数并启动 FMHA——0.35 µs，零中间内核，零拷贝。',
    },
    {
      type: 'stats',
      title: '仓库工人不再开封重数',
      sub: '有效 token 数主机上早就有，直接递清单',
      stats: [
        { v: '499.7µs', desc: '→ 0.352µs（p50）', sub: 'all-to-all 结束 → FMHA 启动的间隔；p95 698.3µs → 0.384µs' },
        { v: '244,925', desc: '→ 0', sub: '中间 GPU 活动总数（≈25 个/次调用）' },
        { v: '97,970', desc: '→ 0', sub: 'D2H 拷贝总数（恰好 10 次/次调用）' },
      ],
      footnote: '⚠️ 这是间隔级测量，不是端到端加速；端到端用帧 SHA256 与基线完全一致验证正确性。上游 4x B300 Nsight 采样，非 cookbook 跑分。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '含主机契约逐条校验、多文档连续批处理与 #5543 rebase 关系',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（系列 Blog 3）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — H3 系列第 3 篇，无需新开关即自动生效' },
      ],
      disclaimer: '本文为 MiniMax-H3 优化系列 Blog 3（非 GPU 热路径主题旗舰篇）。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '删掉每层500微秒等待',
    body: [
      'MiniMax-H3 在 B200/B300 上默认走 TRTLLM 注意力，但每次注意力调用前，GPU 都要把 padding 掩码在卡上重算一遍：求和、比对、搜边界、再回读 10 次标量——每次回读都强迫 GPU 停下来等 CPU📦',
      '',
      '就像仓库工人收到带打印清单的托盘，却坚持把每个箱子开封重数，而且每层网络、每次调用都重数一遍😅',
      '',
      'PR #6542 一句话：打包的人本来就知道有效 token 数（主机上现成的整数），直接递清单就行——新增 PackedPaddingMetadata，TRTLLM 校验后用普通切片裁剪 Q/K/V，不再读任何设备标量。',
      '',
      '4x B300 实测（上游 Nsight，starship 1344×768·243帧·49步）：',
      '⚡ 间隔 p50：499.683µs → 0.352µs（p95 698.3µs → 0.384µs）',
      '🧮 中间 GPU 活动 244,925 → 0（约25个/次）',
      '📤 D2H 拷贝 97,970 → 0（10次/次）',
      '',
      '诚实说明：这是 all-to-all 结束到 FMHA 启动的间隔级测量，不是端到端加速；端到端以帧 SHA256 与基线完全一致验证正确性。',
      '',
      '它还顺手修了 B200 默认后端直接报错的问题（issue #6358）。无需新开关，选 TRTLLM_ATTN 自动生效；要任意逐 token 掩码仍需 CUDNN_ATTN/TORCH_SDPA。',
      '',
      '📘 完整拆解：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'vLLM', '推理优化', '视频生成'],
  },
});
