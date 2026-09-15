// Xiaohongshu note — blog post (zh):
// _posts/2026-09-15-understanding-pr-5779-5990-minimax-h3-kernels.zh.md
// Captions lifted from the post's zh figure alt text; single-run and
// author-reported caveats kept in both the stats footnote and the body.

const FIG = 'blog/assets/figures/pr-5779-5990-minimax-h3-kernels';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#4f46e5', // feature: kernels palette from blog/_config.yml
  colorDeep: '#4338ca',
  colorDarkest: '#3730a3',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '换内核快14%？' },
        '再融合，只赚3%',
      ],
      sub: 'MiniMax-H3 内核篇：TRTLLM 成为默认后端\nQ/K 归一化+RoPE 融合进单个 kernel',
      chips: [
        { v: '14.1%', label: '一次配对实测' },
        { v: '−3.18%', label: '融合 prologue' },
        { v: '58816', label: 'packed 对齐行数' },
      ],
    },
    {
      type: 'figure',
      title: '装货板的故事',
      sub: 'PR #5779 · packed 序列 × TRTLLM',
      imgAbs: `${ROOT}/${FIG}/fig1-packed-trim.svg`,
      tag: '图1',
      caption: 'H3 把 58,758 个真实 token 装进 58,816 行的对齐张量，末尾 58 行是填充。TRTLLM 以前见到带 mask 的货板直接拒收；现在读货单（cu_seqlens 元数据）、只对真实 token 做 attention，再把形状补回去交给 Ulysses 通信。顺带修掉两个坑：填充行进了 SAGE 量化会输出非有限值；14-token 的 token refiner 比一个量化 block 还短，自动走 dense。',
    },
    {
      type: 'figure',
      title: '两道工序并成一个',
      sub: 'PR #5990 · fused QK norm + RoPE',
      imgAbs: `${ROOT}/${FIG}/fig2-qk-fusion.svg`,
      tag: '图2',
      caption: '每次 attention 前，Q 和 K 各要做"归一化 + 旋转"两道小工序。以前是一串小 kernel，中间结果反复写显存；现在一个 Triton kernel 在寄存器里全做完——一读一写。每个 DiT block、每个 denoise forward、一段视频 49 遍，每遍都省一点。',
    },
    {
      type: 'stats',
      title: '数字与诚实说明',
      sub: '内核赢面 ≠ 端到端赢面',
      stats: [
        { v: '14.1%', desc: 'TRTLLM vs FA4 · diffusion', sub: '4×B300 配对实测（seed 1101，排除编译预热）' },
        { v: '−3.18%', desc: '融合 Q/K prologue', sub: 'steady denoising 109.687s → 106.205s' },
        { v: '27.10dB', desc: 'PSNR：TRTLLM vs FA4', sub: '连 dense 后端之间输出也不是逐位一致（SSIM 0.8880）' },
      ],
      footnote: '⚠️ 全部为上游作者报告数字，非 cookbook 基准。14.1% 是单次配对运行，recipe 的稳定 A/B 说 dense 两者相差 2% 以内——设默认的真正理由是 TRTLLM 独占 Skip-Softmax/SAGE；3.18% 是单 seed、单请求。SAGE/Skip-Softmax 本身是有损开关。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '含 14-token 站点保护、timestep 发布与怎么选的决策卡',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（系列 Blog 5）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — H3 系列第 5 篇，内核主题核心篇' },
      ],
      disclaimer: '本文为 MiniMax-H3 优化系列 Blog 5（内核主题核心篇）。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '换内核快14%？再融合只赚3%',
    body: [
      '给视频生成模型做内核优化，到底能赚多少？MiniMax-H3 这周给了两个数字：一个很响，一个很诚实。',
      '',
      '第一个：TRTLLM 注意力内核以前直接拒收 H3 的 packed 序列（末尾有 58 行填充）。PR #5779 教它读"货单"——只对 58,758 个真实 token 做 attention，形状再补回去。修完顺带成为数据中心 Blackwell 的默认后端。一次配对实测：比 FA4 快 14.1%（diffusion 阶段）🚛',
      '',
      '第二个：每次 attention 前，Q/K 要过"归一化+旋转"两道小工序，中间结果反复写显存。PR #5990 把它们融成一个 Triton kernel，一读一写。结果：steady denoising 只降 3.18%🔍',
      '',
      '为什么融合赚得少？因为 attention 本身才是大头，前置工序只是小额税费——内核局部赢面再大，也不会线性传导到端到端。这是所有"我们 kernel 快了 X 倍"新闻该问的第一句。',
      '',
      '诚实说明：以上都是上游作者报告的数字；14.1% 是单次配对（稳定 A/B 说 dense 两者 2% 以内，默认的真正理由是 TRTLLM 独占稀疏/量化两扇门）；3.18% 是单 seed 单请求；连 dense 后端间输出都有差异（PSNR 27.10dB）。',
      '',
      '📘 完整拆解：主页简介直达（系列 Blog 5）',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'vLLM', '推理优化', '视频生成'],
  },
});
