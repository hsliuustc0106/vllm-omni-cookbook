// Xiaohongshu note — blog post (zh):
// _posts/2026-09-15-understanding-pr-6516-sensenova-u15-distilled-lora.zh.md
// Captions lifted from the post's zh figure alt text; PR-review-thread provenance kept.

const FIG = 'blog/assets/figures/pr-6516-sensenova-u15-distilled-lora';

export default (ROOT) => ({
  brand: {
    series: 'vLLM-Omni',
    seriesZh: '源码精读',
    kicker: '源码精读 · PR 分析',
    github: 'vllm-omni-cookbook',
    site: 'hsliuustc0106.github.io/vllm-omni-cookbook',
  },
  color: '#c026d3', // feature: lora palette from blog/_config.yml
  colorDeep: '#a21caf',
  colorDarkest: '#8616a7',

  cards: [
    {
      type: 'cover',
      lines: [
        { text: '加载成功的LoRA' },
        '其实没生效',
      ],
      sub: 'SenseNova-U1.5 × PR #6516：一个子串 bug\n和一条被拧住的自回归解码',
      chips: [
        { v: '0.48%', label: '带不带LoRA的差别' },
        { v: '2×高度', label: 'bug拼出的delta' },
        { v: '~0.9s', label: '1024²八步出图' },
      ],
    },
    {
      type: 'figure',
      title: '一个子串，两倍高度',
      sub: 'PR #6516 · 蒸馏 LoRA 静默失效',
      imgAbs: `${ROOT}/${FIG}/fig1-lora-substring-bug.svg`,
      tag: '图1',
      caption: '两条堆叠映射规则同时命中：".qkv_proj" 是 ".qkv_proj_mot_gen" 的子串，融合 delta 高度变成参数的两倍，适配器从未生效。修复：按名字尾部匹配、首个命中即停——588 个键真正融进 168 个参数。',
    },
    {
      type: 'figure',
      title: 'graph 要静态形状',
      sub: '分页 KV 是唯一出路',
      imgAbs: `${ROOT}/${FIG}/fig2-paged-decode.svg`,
      tag: '图2',
      caption: 'CUDA graph 需要静态形状：torch.cat 每步变形状、无法捕获；垫桶+mask 静态但每步 7.96–11.55ms vs 无 mask 1.03ms，被否；分页缓存保持桶宽缓冲，flash_attn_varlen_func 经 seqused_k 只读有效前缀——一次捕获服务整桶。',
    },
    {
      type: 'stats',
      title: '修好之后有多快',
      sub: '1× A800 · 1024² · think 开 · P50',
      stats: [
        { v: '16.1→3.5s', desc: 'think 解码', sub: '50 步端到端 27.7s→13.9s；换 8 步 LoRA 总共 4.5s' },
        { v: '97.7%', desc: 'GPU 利用率', sub: 'graph 前空闲 3.19s → 后 0.12s；kernel 时间持平' },
        { v: '−67.9%', desc: 'kernel 发射数', sub: '24,244→7,782；主机开销 37.3%→2.3%' },
      ],
      footnote: '⚠️ 数字均来自 PR 评审线程实测（作者 1×A800 / 评审人 L20X、H200），非本 cookbook 基准。首个请求多 ~0.7s 编译；graph 捕获按请求 30–34ms；只在 80GB 级显卡验证过。',
    },
    {
      type: 'figure',
      title: '分页缓存≠更快',
      sub: 'graph 才是买时间的人',
      imgAbs: `${ROOT}/${FIG}/fig3-decode-arms.svg`,
      tag: '图3',
      caption: '三条解码臂各差一个东西：只换分页缓存（A→B）毫无变化；包上 CUDA graph（B→C）后空闲从 3,186.7ms 塌缩到 120.0ms——kernel 时间持平，省的全是主机派发。分页缓存买到形状，graph 买到时间。',
    },
    {
      type: 'end',
      title: '完整拆解在这里',
      sub: '含"试过又放弃"的实验表、决策卡与 H200 验证',
      paths: [
        { title: '小红书主页 → 简介', desc: '点开简介里的链接直达这篇' },
        { title: 'GitHub 搜索框输入', desc: 'vllm-omni-cookbook — PR 分析系列' },
      ],
      disclaimer: '本文对应 vLLM-Omni PR #6516（2026-09-01 合并）。\nvLLM-Omni 社区出品 · 只讲代码背后发生的事。',
    },
  ],

  note: {
    title: '加载成功的LoRA，其实没生效',
    body: [
      '商汤 SenseNova-U1.5-8B-MoT 进 vLLM-Omni（PR #6516）。一个"模型支持"PR 的自我升级：',
      '',
      '🔧 bug1：官方 8 步蒸馏 LoRA 加载打印成功，实际一个权重没改——main 上带不带 --lora-path 只差 0.48%。根因是子串匹配：".qkv_proj" 是 ".qkv_proj_mot_gen" 的子串，两条映射规则同时命中，拼出的 delta 是参数高度的两倍。改成按尾部匹配、首个命中即停：588 个键真正融进 168 个参数。',
      '',
      '🐌 bug2：评审用 Nsight 一测，think 解码 86% 的空闲根本不在任何 CUDA 调用里——纯 Python 派发。解法：分页 KV 缓存（flash_attn_varlen 的 seqused_k）让形状静态化，再包一层 CUDA graph。',
      '',
      '📊 A800 实测：think 解码 16.1s→3.5s；GPU 利用率 62.7%→97.7%；kernel 发射 −67.9%；1024² 八步出图约 0.9s。',
      '',
      '⚠️ 诚实说明：数字全部来自 PR 评审线程（作者 A800 / 评审人 L20X、H200），非独立基准；graph 捕获按请求发生、think 文本跨后端可能差一个 token；只在 80GB 级显卡验证过。',
      '',
      '💡 记得：这个 LoRA 配 CFG 1.0，配 4.0 会过曝色带化。',
      '',
      '📘 完整拆解：主页简介直达',
      '🔍 GitHub 搜：vllm-omni-cookbook',
      '',
    ].join('\n'),
    tags: ['大模型', 'AI技术分享', '开源项目', 'vLLM', '推理优化', 'LoRA'],
  },
});
