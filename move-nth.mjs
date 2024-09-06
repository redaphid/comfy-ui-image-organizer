#!/usr/bin/env node
import * as fs from 'fs/promises';
import { exec as execCallBack } from 'child_process';
import { promisify, parseArgs } from 'util';
import path from 'path';

const exec = promisify(execCallBack);

const getTimestamp = async (imagePath) => {
  // get file modified date
  const { mtime } = await fs.stat(imagePath);
  console.log('mtime:', mtime);
  return new Date(mtime).toISOString();
};

const getModelDirName = async (imagePath) => {
  // exec exiftool -p prompt imagePath
  const exifCommand = `exiftool -b -prompt "${imagePath.trim()}"`;
  console.log('exifCommand:', exifCommand);
 try {
  const { stdout } = await exec(exifCommand, { maxBuffer: 1024 * 1024 * 1024 });
  const flow = JSON.parse(stdout);
  const models = [];
  for (const nodeId in flow) {
    const node = flow[nodeId];
    if (node.inputs.ckpt_name) {
      models.push(node.inputs.ckpt_name.split('.')[0]);
    }
  }
  return models.join('/');
} catch (error) {
  return "unknown";
}
};

const getDestDir = (imagePath, timestamp, modelDirName) => {
  console.log('getDestDir:', imagePath, timestamp, modelDirName);
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
      findImageFiles(filePath);
      continue;
    }
    console.log("is file", filePath);

    const [timestamp,modelDirName] = await Promise.allSettled([getTimestamp(filePath),getModelDirName(filePath)]);
    if(timestamp.status === 'rejected' || modelDirName.status === 'rejected'){
      console.log('Error getting timestamp or modelDirName', timestamp, modelDirName);
      continue;
    }
    const destDir = getDestDir(filePath, timestamp.value, modelDirName.value);
    console.log({ destDir });

    // Move the file to the new directory
    await moveFile(filePath, destDir);
  }
};

const main = async () => {
  // finds all files in the input directory, takes the number of files and the number of directories, and logs them
  let {
    values: {inputDir, strideLength, outputDir},
  } = parseArgs({
    options: {
      inputDir: { type: 'string', short: 'i' },
      outputDir: { type: 'string', short: 'o'},
      strideLength: { type: 'string', short: 's'},
    },
  });
  console.log({ inputDir, outputDir, strideLength });
  strideLength = parseInt(strideLength, 10);
  console.log({ strideLength });
};

main();
