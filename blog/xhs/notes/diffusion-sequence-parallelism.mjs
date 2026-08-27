// Xiaohongshu note — blog post:
// _posts/2026-08-17-understanding-diffusion-sequence-parallelism.md
// All speedup figures are upstream-doc claims; copy keeps that caveat.

const FIG = 'blog/assets/figures/diffusion-sequence-parallelism';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · 特性深潜',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#2563eb', // feature: parallelism palette from blog/_config.yml
  colorDeep: '#1e40af',
  colorDarkest: '#172554',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '一张图几万个token' },
        '4种切法分给多卡',
      ],
      sub: 'vLLM-Omni 序列并行：Ulysses / Ring / Hybrid / AllGather-KV\n官方口径最高 <b>×3.6</b>',
      chips: [
        { v: '×3.6', label: '最高提速（官方文档）' },
        { v: '23', label: '模型家族已接好' },
        { v: '2–8卡', label: '适用规模' },
      ],
    },
    {
      type: 'figure',
      title: '切法总览：一组卡，两种分法',
      sub: '每个 SP 组 = Ulysses 组 × Ring 组',
      imgAbs: `${ROOT}/${FIG}/fig1-topology.png`,
      tag: '图1',
      caption: '实线框是连续切分的 Ulysses 组（按人头分队），虚线箭头是跨步组成的 Ring 组（按花色传圈）。u2×r2、u4×r1、u1×r4 只是同一套机制的三种预设。',
    },
    {
      type: 'figure',
      title: 'Ulysses：用“头”换“序列”',
      sub: '一次 all-to-all，互换两个维度',
      imgAbs: `${ROOT}/${FIG}/fig2-ulysses.png`,
      tag: '图2',
      caption: '每张卡先拿一小段 token 的全部注意力头；all-to-all 之后变成 全序列 × 少数头，kernel 当普通输入算，算完再换回来。Q/K/V/O 每层全员搬家两趟 —— 大图 + NVLink 首选。',
    },
    {
      type: 'figure',
      title: 'Ring：Q 不动，K/V 击鼓传花',
      sub: '通信提前发出，藏在计算背后',
      imgAbs: `${ROOT}/${FIG}/fig3-ring.png`,
      tag: '图3',
      caption: 'K/V 块每步往前传一张卡，每步的部分结果用 online softmax 合并成最终输出。Q 从头到尾不出门 —— 超长序列、显存紧张时的选择。',
    },
    {
      type: 'figure',
      title: 'Hybrid 与 AllGather-KV',
      sub: '大规模合体 & 极简兜底',
      imgAbs: `${ROOT}/${FIG}/fig4-hybrid-allgather.png`,
      tag: '图4',
      caption: 'Hybrid：先组内 all-to-all 换到全序列，再跨组传一圈 —— 8 卡以上避免把“头”切太碎。AllGather-KV：两次 all-gather 拼出完整 K/V，Q 原地算，零整除约束，最简单。',
    },
    {
      type: 'stats',
      title: '怎么选，一张表',
      sub: '通信量、约束、适用场景',
      stats: [
        { v: '大图', desc: 'Ulysses', sub: 'Q/K/V 每层全员两趟；头数需整除（30 头模型走 experimental UAA）' },
        { v: '长序列', desc: 'Ring', sub: 'K/V 一跳/步且与计算重叠；环内分片等长' },
        { v: '>8卡', desc: 'Hybrid u×r', sub: '先 Ulysses 后 Ring；要求全局序列长度一致' },
        { v: '小KV', desc: 'AllGather-KV', sub: '两次 all-gather，Q 本地；仅非因果注意力' },
      ],
      footnote: '⚠️ 提速 1.5×–3.6× 为上游文档口径（2–8 卡、大图/视频），本篇未独立复测；<1024px 短序列官方建议老实单卡。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '这篇的每一张图都是网页里活着的动图 —— 相位步进器、会转圈的 Ring，浏览器里直接玩',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（动图版）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — 序列并行篇，4 种策略的代码与约束矩阵都在' },
      ],
      disclaimer: '提速数字来自上游官方文档，cookbook 尚未独立复测。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '视频生成提速3.6倍：序列并行4式',
    body: [
      '2048×2048 的一张图，切成 patch 喂给扩散模型就是几万个 token；几秒视频再多一个量级。Attention 开销随 token 数平方涨，单卡直接 OOM💀',
      '',
      '但有个救命的结构性事实：DiT 里只有 attention 是“全局”的，其余全是逐 token 的独立计算。所以把序列切成 S/P 份，除 attention 外一切照旧 —— 只需要约定好 attention 怎么交换信息。',
      '',
      'vLLM-Omni 给了 4 种可插拔姿势：',
      '🔹 Ulysses：all-to-all 用“头”换“序列”，kernel 看到全序列；大图+NVLink 首选',
      '🔹 Ring：Q 不动，K/V 击鼓传花，通信藏在计算后面；超长序列首选',
      '🔹 Hybrid：先 Ulysses 后 Ring，8 卡以上不把头切太碎',
      '🔹 AllGather-KV：两次 all-gather，Q 原地算，零约束最简单',
      '',
      '官方文档口径：大图/视频 2–8 卡提速 1.5×–3.6×，23 个模型家族已接好（本篇未独立复测⚠️）',
      '',
      '开法简单到离谱：vllm serve --usp 2 就是 Ulysses，--ring 2 就是 Ring，一起给就是 Hybrid✅',
      '',
      '小图（<1024px）老实用单卡 —— Ring 的绕圈开销会反噬。',
      '',
      '📘 完整拆解（含可交互动图）：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'AIGC', '深度学习', 'GPU', '并行计算'],
  },
});
