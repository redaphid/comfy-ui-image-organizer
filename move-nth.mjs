#!/usr/bin/env node
import * as fs from 'fs/promises';
import { parseArgs } from 'util';
import path from 'path';

// Find all files, optionally recursively if recursive flag is set
const findFiles = async (dirPath, recursive) => {
  const files = await fs.readdir(dirPath, { withFileTypes: true });
  let allFiles = [];

  await Promise.all(files.map(async (file) => {
    const filePath = path.join(dirPath, file.name);
    if (file.isDirectory() && recursive) {
      const subFiles = await findFiles(filePath, recursive);
      allFiles = allFiles.concat(subFiles);
    } else if (!file.isDirectory()) {
      const { mtime } = await fs.stat(filePath);
      allFiles.push({ path: filePath, timestamp: mtime });
    }
  }));

  return allFiles;
};

// Copy or move the file to the specified destination
const processFile = async (source, destination, move) => {
  const destDir = path.dirname(destination);
  await fs.mkdir(destDir, { recursive: true });

  if (move) {
    await fs.rename(source, destination);
    console.log(`Moved ${source} to ${destination}`);
  } else {
    await fs.copyFile(source, destination);
    console.log(`Copied ${source} to ${destination}`);
  }
};

// Process every <stride> file into numbered folders
const processNthFiles = async ({ input, output, stride, recursive, move }) => {
  const files = await findFiles(input, recursive);

  // Sort files by creation date
  files.sort((a, b) => a.timestamp - b.timestamp);

  // Copy or move files into subfolders based on stride
  await Promise.all(files.map(async (file, index) => {
    const folderNumber = (index % stride) + 1;
    const destination = path.join(output, folderNumber.toString(), path.basename(file.path));
    await processFile(file.path, destination, move);
  }));
};

// Main function to parse arguments and execute processNthFiles
const main = async () => {
  const {
    values: { inputDir, strideLength, outputDir, recursive, move },
  } = parseArgs({
    options: {
      inputDir: { type: 'string', short: 'i' },
      outputDir: { type: 'string', short: 'o' },
      strideLength: { type: 'string', short: 's' },
      recursive: { type: 'boolean', short: 'r', default: false },
      move: { type: 'boolean', short: 'm', default: false },
    },
  });

  // Validate input arguments
  if (!inputDir) {
    throw new Error("Missing required argument: --inputDir (-i). You must provide the input directory.");
  }
  if (!outputDir) {
    throw new Error("Missing required argument: --outputDir (-o). You must provide the output directory.");
  }
  if (!strideLength || isNaN(parseInt(strideLength, 10)) || parseInt(strideLength, 10) <= 0) {
    throw new Error("Invalid or missing argument: --strideLength (-s). You must provide a valid positive number for the stride.");
  }

  const stride = parseInt(strideLength, 10);

  // Start the file copying/moving process
  await processNthFiles({ input: inputDir, output: outputDir, stride, recursive, move });
};

main().catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1); // Exit with failure status code
});
