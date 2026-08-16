#!/usr/bin/env ruby
# frozen_string_literal: true

# Blog post lint: every blog/_posts/*.md must carry a `summary` front-matter
# field of at most 240 characters (SEO description + index card teaser).
# Minimal reimplementation of the convention vllm.ai enforces in CI.
#
# Usage: ruby scripts/check-blog-summaries.rb [posts_dir]

require "yaml"

posts_dir = ARGV[0] || File.join(__dir__, "..", "blog", "_posts")
posts = Dir.glob(File.join(posts_dir, "*.md")).sort

if posts.empty?
  warn "no posts found under #{posts_dir}"
  exit 1
end

config = YAML.safe_load(File.read(File.join(__dir__, "..", "blog", "_config.yml")),
                        permitted_classes: [Date, Time], aliases: false) || {}
feature_slugs = (config["features"] || []).map { |f| f["slug"].to_s.downcase }

bad = []
all_tags = Hash.new { |h, k| h[k] = [] } # downcased tag -> [original, file]
posts.each do |path|
  begin
    front = File.read(path).split(/^---\s*$/)[1]
    meta = front ? YAML.safe_load(front, permitted_classes: [Date, Time], aliases: false) : nil
  rescue StandardError => e
    bad << "#{File.basename(path)}: unparseable front matter (#{e.class}: #{e.message})"
    next
  end

  summary = meta.is_a?(Hash) ? meta["summary"] : nil
  if summary.nil? || summary.to_s.strip.empty?
    bad << "#{File.basename(path)}: missing `summary` front matter"
  elsif summary.length > 240
    bad << "#{File.basename(path)}: summary is #{summary.length} chars (max 240)"
  end

  (meta.is_a?(Hash) ? meta["tags"] : []).to_a.each do |tag|
    tag = tag.to_s
    all_tags[tag.downcase] << [tag, File.basename(path)]
    if feature_slugs.include?(tag.downcase)
      bad << "#{File.basename(path)}: tag `#{tag}` duplicates a site.features slug — classify via `feature:` front matter instead"
    end
  end
end

all_tags.each do |down, variants|
  originals = variants.map(&:first).uniq
  if originals.size > 1
    bad << "tag casing drift: #{originals.join(" vs ")} (#{variants.map(&:last).uniq.join(", ")}) — use one canonical form"
  end
end

if bad.any?
  warn bad.join("\n")
  exit 1
end

puts "#{posts.size} post(s) OK (summary <= 240 chars)"
