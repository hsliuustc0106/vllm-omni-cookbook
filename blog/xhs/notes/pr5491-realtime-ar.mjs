// Xiaohongshu note — blog post:
// _posts/2026-08-16-pr5491-realtime-ar-diffusion.md
// All numbers are PR-author-reported; copy keeps that + the RNG-order story.

const FIG = 'blog/assets/figures/pr5491';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#db2777', // feature: realtime_ar_diffusion palette from blog/_config.yml
  colorDeep: '#9d174d',
  colorDarkest: '#831843',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '视频生成也能' },
        '边玩边转向',
      ],
      sub: '世界模型变成“可操控游戏”：每个 tick 长一小段\n你的操作在下一块生效',
      chips: [
        { v: '2.66s', label: '稳态 tick · H200' },
        { v: '10', label: '每轮可交互 tick' },
        { v: '81帧', label: '480×832 输出' },
      ],
    },
    {
      type: 'figure',
      title: '一次请求 = 一个 AR 块',
      sub: 'tick 式推进，会话状态跨请求活着',
      imgAbs: `${ROOT}/${FIG}/fig1.png`,
      tag: '图1',
      caption: '会话控制面把你的操作排队、冻结成 typed tick；引擎照常生成；返回的元数据四个身份全部对上才提交。分页 KV 把历史留在会话里，下一个 tick 接着长。',
    },
    {
      type: 'figure',
      title: 'Fail-closed：错就明说',
      sub: '宁可重开一个世界，不带病续跑',
      imgAbs: `${ROOT}/${FIG}/fig2.png`,
      tag: '图2',
      caption: '任何 runner/元数据/提交失败：会话立刻 FAILED、释放 KV、拒绝原地重试 —— 只有显式 reset 才能重开。连“关闭失败”都会留下可重试的墓碑，绝不假装成功。',
    },
    {
      type: 'figure',
      title: '分页 KV：会话的记忆',
      sub: '锚点 + 最近窗口 + 当前块',
      imgAbs: `${ROOT}/${FIG}/fig3.png`,
      tag: '图3',
      caption: '驻留策略：sink 9 块 + 最近窗口 18 块 + 当前块 3 帧，提交后剪枝中间。与直连 attention 七块重放 bit-exact —— 分页只是省显存，不改结果。',
    },
    {
      type: 'stats',
      title: '数字说话（作者实测 H200）',
      sub: '480×832 · 4 步 DMD · LingBot World 2.0',
      stats: [
        { v: '5.19s', desc: '首个 tick', sub: '含会话建立；稳态平均 2.66s/块' },
        { v: '10', desc: '每轮 tick 上限', sub: '117 帧图像视野 ÷ 每块 3 帧；第 11 个 tick 提前拒绝' },
        { v: '0', desc: '分页 vs 直连误差', sub: '7 块 / 81 帧重放 bit-exact（B200 验证）' },
      ],
      footnote: '⚠️ 数字为 PR 作者报告。对齐官方 RNG 顺序后 PSNR 15.05 → 21.44 dB —— 与官方实现的差异主要是“什么时候抽噪声”，不是 kernel 算错。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '还配了一个可缩放的交互式架构走查（tick 时间线 + 不变量网格）',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇（交互架构图版）' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — 实时 AR 扩散篇，事件 schema 和限制清单都在' },
      ],
      disclaimer: 'PR #5491 已合并；当前为实验特性（example + recipe），公开 HTTP API 在 draft #5527。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '边生成边转向：可操控的世界模型',
    body: [
      '世界模型的终极形态是什么？是你看到第 27 帧，再决定第 30 帧的镜头往哪转🎮',
      '',
      '但“可交互”对推理引擎是个大难题：请求与请求之间，KV 历史、RNG 状态、相机位姿都得活着。请求级的引擎没地方放这些记忆，所以以前只能一次性生成整段视频。',
      '',
      'vLLM-Omni 的 PR #5491 引入了“会话”：',
      '🔹 一次请求 = 一个 AR 块（3 个 latent 帧），tick 式推进',
      '🔹 你的操作（转向/新 prompt）只在块边界生效，排队冻结进下一个 tick',
      '🔹 分页 KV 把历史留在会话里；失败立刻 FAILED，绝不带病续跑',
      '',
      '实测（作者报告，1×H200）：',
      '⚡ 首个 tick 5.19s，稳态 2.66s/块',
      '🎯 每轮最多 10 个 tick（视野上限），reset 开新世界',
      '🧮 分页 KV vs 直连：7 块重放 bit-exact',
      '',
      '最有意思的发现：和官方脚本对比，只把噪声抽取顺序对齐，PSNR 就从 15 涨到 21 dB —— 差异原来主要是“什么时候抽噪声”，不是算错😅',
      '',
      '目前是实验特性（example + recipe），公开 API 在 draft #5527。',
      '',
      '📘 完整拆解（含交互式架构图）：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', '世界模型', 'AIGC', '视频生成', '实时生成'],
  },
});
