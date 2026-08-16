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

bad = []
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
end

if bad.any?
  warn bad.join("\n")
  exit 1
end

puts "#{posts.size} post(s) OK (summary <= 240 chars)"
