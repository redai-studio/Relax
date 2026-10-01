#!/usr/bin/env node

/**
 * Fix chunk filenames to avoid .md. pattern that triggers WAF 403.
 * Renames files and updates ALL references in HTML, JS, and JSON files.
 *
 * VitePress generates *.md.HASH.lean.js files for page data; these are NOT
 * controlled by rollupOptions.chunkFileNames and must be fixed post-build.
 * References to these files exist in both HTML and JS bundles.
 */

const fs = require('fs');
const path = require('path');

const distDir = path.join(__dirname, '.vitepress/dist');

// Walk directory and collect files matching a predicate
const walkFiles = (dir, predicate) => {
  const result = [];
  const walk = (currentPath) => {
    const entries = fs.readdirSync(currentPath, {withFileTypes : true});
    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (predicate(entry.name)) {
        result.push(fullPath);
      }
    }
  };
  walk(dir);
  return result;
};

// Find all files whose name contains the .md. pattern
const findFilesWithMdPattern = (dir) =>
    walkFiles(dir, (name) => name.includes('.md.'));

// Build a mapping: old filename → new filename (replace every .md. with -)
const createRenameMap = (files) => {
  const map = {};
  for (const file of files) {
    const filename = path.basename(file);
    const newFilename = filename.replace(/\.md\./g, '-');
    if (filename !== newFilename) {
      map[filename] = newFilename;
    }
  }
  return map;
};

// Rename physical files on disk
const renameFiles = (files, renameMap) => {
  for (const file of files) {
    const dir = path.dirname(file);
    const filename = path.basename(file);
    const newFilename = renameMap[filename];
    if (!newFilename)
      continue;
    const newPath = path.join(dir, newFilename);
    try {
      fs.renameSync(file, newPath);
      console.log(`  renamed : ${filename}`);
      console.log(`       -> : ${newFilename}`);
    } catch (err) {
      console.error(`  FAILED  : ${filename} — ${err.message}`);
    }
  }
};

// Update all string occurrences of old filenames in a set of text files.
// Covers HTML, JS (including lean.js bundles), and JSON files.
const updateTextReferences = (renameMap, dir) => {
  const textFiles =
      walkFiles(dir, (name) => name.endsWith('.html') || name.endsWith('.js') ||
                               name.endsWith('.json'));

  let updatedCount = 0;
  for (const filePath of textFiles) {
    let content;
    try {
      content = fs.readFileSync(filePath, 'utf-8');
    } catch {
      continue;
    }

    let updated = false;
    for (const [oldName, newName] of Object.entries(renameMap)) {
      // Escape special regex characters in the filename
      const escaped = oldName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(escaped, 'g');
      if (re.test(content)) {
        content = content.replace(re, newName);
        updated = true;
      }
    }

    if (updated) {
      fs.writeFileSync(filePath, content, 'utf-8');
      console.log(`  updated : ${path.relative(dir, filePath)}`);
      updatedCount++;
    }
  }
  return updatedCount;
};

// Main
try {
  if (!fs.existsSync(distDir)) {
    console.error(`dist directory not found: ${distDir}`);
    console.error('Run "npm run docs:build" first.');
    process.exit(1);
  }

  console.log('Fixing .md. chunk filenames to bypass WAF...\n');

  const files = findFilesWithMdPattern(distDir);
  if (files.length === 0) {
    console.log('No files with .md. pattern found — nothing to do.');
    process.exit(0);
  }

  console.log(`Found ${files.length} file(s) to rename:\n`);
  const renameMap = createRenameMap(files);

  console.log('-- Renaming files --');
  renameFiles(files, renameMap);

  console.log('\n-- Updating references in HTML / JS / JSON --');
  const updatedCount = updateTextReferences(renameMap, distDir);

  console.log(`\nDone. Renamed ${files.length} file(s), updated references in ${
      updatedCount} file(s).`);
} catch (err) {
  console.error('Error:', err.message);
  process.exit(1);
}
