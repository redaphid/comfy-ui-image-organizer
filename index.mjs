#!/usr/bin/env node
import * as fs from 'fs/promises';
import { exec as execCallBack } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const exec = promisify(execCallBack);

const getTimestamp = async (imagePath) => {
  // get file modified date
  const { mtime } = await fs.stat(imagePath);
  return new Date(mtime).toISOString();
};

const getModelDirName = async (imagePath) => {
  // exec exiftool -p prompt imagePath
  const { stdout } = await exec(`exiftool -b -prompt ${imagePath.trim()}`, { maxBuffer: 1024 * 1024 * 1024 });
  const flow = JSON.parse(stdout);
  const models = [];
  for (const nodeId in flow) {
    const node = flow[nodeId];
    if (node.inputs.ckpt_name) {
      models.push(node.inputs.ckpt_name.split('.')[0]);
    }
  }
  return models.join('/');
};

const getDestDir = (imagePath, timestamp, modelDirName) => {
  const newFileName = new Date(timestamp).getTime() + '.png';
  const dateDirName = new Date(timestamp).toISOString().split('T')[0];
  const destDir = path.join('./out',dateDirName, modelDirName, newFileName);
  return destDir;
};

const moveFile = async (source, destination) => {
  // Create the destination directory if it doesn't exist
  const destDir = path.dirname(destination);
  await fs.mkdir(destDir, { recursive: true });

  // Move the file
  await fs.rename(source, destination);
  console.log(`Moved ${source} to ${destination}`);
};

const findImageFiles = async (dirPath) => {
  console.log('findImageFiles: looking at', dirPath);
  const files = await fs.readdir(dirPath);
  for (const file of files) {
    // if it's a directory, recurse
    const filePath = path.join(dirPath, file);
    const stat = await fs.stat(filePath);
    if (stat.isDirectory()) {
      console.log('filePath:', filePath, 'isDirectory');
      await findImageFiles(filePath);
      continue;
    }
    console.log("is file", filePath);
    // if it starts with %, then ignore it for now
    if (file.includes('%')) {
      console.log('ignoring', file);
      continue;
    }
    // if the file doesn't end with .png, ignore it
    if (!file.endsWith('.png')) {
      console.log('ignoring', file);
      continue;
    }
    const timestamp = await getTimestamp(filePath);
    const modelDirName = await getModelDirName(filePath);
    const destDir = getDestDir(filePath, timestamp, modelDirName);
    console.log({ destDir });

    // Move the file to the new directory
    await moveFile(filePath, destDir);
  }
};

const main = async () => {
  await findImageFiles('./in/');
};

main();
