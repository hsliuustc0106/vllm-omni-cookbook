// Pilot xiaohongshu note — blog post:
// _posts/2026-08-16-understanding-pr-6162-svdquant-w4a4-blackwell.md
//
// Copy follows the XHS calibration: layman reader, number-forward hooks,
// eli5 analogies, honest caveats kept in (draft PR, author-reported numbers).

const FIG = 'blog/assets/figures/pr-6162-svdquant';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
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
        '一个离群值',
        { text: '毁掉整张量化网格' },
      ],
      sub: '视频生成大模型想全上 4bit？\n先过这一关 — <b>SVDQuant</b> 用 <b>1.6% 的算力</b>把它救回来了',
      chips: [
        { v: '−34%', label: '峰值显存' },
        { v: '×1.27', label: '端到端提速' },
        { v: 'W4A4', label: '权重+激活全 4bit' },
      ],
    },
    {
      type: 'figure',
      title: '翻车现场：一根通道带崩全场',
      sub: 'NVFP4 只有 16 个刻度，diffusion 权重偏偏到处是尖刺',
      imgAbs: `${ROOT}/${FIG}/fig1.png`,
      tag: '图1',
      caption: '左边：整个网格共用一个全局 scale，一个离群值就把大家的精度拖垮。右边：每 16 个元素一组块 scale，损坏被隔离在局部 —— 但这只是及格线，不是满分。',
    },
    {
      type: 'figure',
      title: '抢救方案：把矩阵拆成两半',
      sub: '低秩部分收留离群值，剩下的放心转 4bit',
      imgAbs: `${ROOT}/${FIG}/fig2.png`,
      tag: '图2',
      caption: '每层权重平滑后 SVD 分解：W ≈ L + R。低秩部分 L 保持 BF16，专职收留离群值；残差 R 长得平坦，量化几乎不疼。推理时两条路加起来 —— r=4 时量化误差直接砍半再砍半。',
    },
    {
      type: 'figure',
      title: '草案的多卡分发，合并时被砍',
      sub: '磁盘上一种格式不变；kernel 大一统留给下一阶段',
      imgAbs: `${ROOT}/${FIG}/fig5.png`,
      tag: '图3',
      caption: '草案曾给每代 GPU 自动选 kernel：数据中心 Blackwell 走 FlashInfer，消费级换 Nunchaku，Hopper 直接拒载。评审后这张矩阵没进主线 —— 合并版第一阶段只在 SM103（B300）上验证，复用 vLLM 现成的 NVFP4 kernel。',
    },
    {
      type: 'stats',
      title: '省了多少，快了多少',
      sub: '作者实测：1× B300 · 5 秒 50 步视频 · seed 固定',
      stats: [
        { v: '−34%', desc: 'worker 峰值显存', sub: '132 GB → 87 GB，单卡能装下了' },
        { v: '×1.27', desc: '端到端耗时', sub: '134.6 s → 106.4 s，去噪阶段 ×1.29' },
        { v: '~1.6%', desc: '低秩分支额外算力开销', sub: '这就是“精度花在刀刃上”的价格标签' },
      ],
      footnote: '⚠️ 数字为作者在完整草案实现上实测（1×B300）；合并版第一阶段是加载路径，融合 kernel 排队 #6493。口径见博客原文。',
    },
    {
      type: 'stats',
      title: '合并了，但只合并了一半',
      sub: '评审把 21 个文件的草案，砍成了 9 个文件的第一阶段',
      stats: [
        { v: '✓', desc: 'checkpoint 契约 + 兼容加载', sub: 'config.json 写一段 quantization_config 即被识别' },
        { v: '✓', desc: 'NVFP4 GEMM + BF16 低秩修正', sub: '复用 vLLM 现成 FlashInfer / CUTLASS / FBGEMM 布局' },
        { v: '✓', desc: 'SM103（B300）已验证', sub: '官方口径：correctness baseline，性能故事未完' },
        { v: '✗', desc: '砍掉，排队 RFC #6493', sub: '原生融合 epilogue · Nunchaku 消费级路径 · 转换器 · 基准 runner' },
      ],
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '博客配了一个可交互可视化 —— 浏览器里跑真的幂迭代 SVD，自己拖着玩',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达博客这篇文章' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — 找到 SVDQuant 那篇，命令和表格都是完整版' },
      ],
      disclaimer: 'PR #6162 已于 2026-08-27 合并（加载版第一阶段）。文中数字为作者在完整草案实现上的报告，未独立复现。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '4bit量化PR合并了，但砍了一半',
    body: [
      '做视频生成的大模型有多大？MiniMax-H3 光一个 DiT，BF16 就要 ~132GB 显存😱 不堆卡根本装不下。',
      '',
      '于是有了 W4A4：权重和激活全部压到 4bit，tensor core 吞吐翻好几倍。但 4bit 只有 16 个刻度 —— 只要有一根“离群”的通道（总是特别大的那一列），它扫过的每个分块都会变形💀',
      '',
      'SVDQuant（PR #6162）的抢救思路很朴素：',
      '🔹 先给权重做个“平滑”，拉平大小差异',
      '🔹 再把矩阵拆成两半：低秩的那半收留离群值（保持 BF16），平坦的另一半放心转 4bit',
      '🔹 推理时两条路加起来，多花的算力只有 ~1.6%',
      '',
      '作者在完整版实现上实测（B300）：',
      '⚡ 端到端 134.6s → 106.4s（×1.27）',
      '📦 峰值显存 132GB → 87GB（−34%）',
      '',
      '🚀 合并进度（8月27日）：进主线了，但砍了一半 ——',
      '✅ checkpoint 格式 + 兼容加载（复用 vLLM 现成 NVFP4 kernel，SM103 已验证）',
      '❌ 自定义融合 kernel、转换器、多代 GPU 分发，排队在 RFC #6493',
      '',
      '带宽受限的 50 个 AdaLN 投影老实不转 4bit，文本编码器保持 BF16 —— 哪里值得省、哪里不能省，这个分寸感我喜欢👍',
      '',
      '📘 完整拆解（含可交互可视化）：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'AIGC', '深度学习', 'vLLM', '推理加速'],
  },
});
