#!/usr/bin/env node
import * as fs from 'fs/promises';
import {exec as execCallBack} from 'child_process';
import {promisify} from 'util';
import path from 'path';
const exec = promisify(execCallBack);

const getDateDirName = async (imagePath) => {
  // get file modified date
  const {mtime} = await fs.stat(imagePath);
  return new Date(mtime).toISOString().split('T')[0];
}

const getModelDirName = async (imagePath) => {
  // exec exiftool -p prompt imagePath
  const {stdout} = await exec(`exiftool -b -prompt ${imagePath}`, {maxBuffer: 1024 * 1024 * 1024});
  const flow = JSON.parse(stdout);
  const models = []
  for(const nodeId in flow) {
    const node = flow[nodeId];
    if(node.inputs.ckpt_name) {
      models.push(node.inputs.ckpt_name.split('.')[0]);
    }
  }
  return models.join('/');
}

const findImageFiles = async (dirPath) => {
  console.log('findImageFiles: looking at', dirPath);
  const files = await fs.readdir(dirPath);
  for(const file of files) {
    // if it's a directory, recurse
    const filePath = path.join(dirPath, file);
    const stat = await fs.stat(filePath);
    if(stat.isDirectory()) {
      console.log('filePath:', filePath, 'isDirectory');
      findImageFiles(filePath);
    }
    console.log("is file", filePath);
    //if it starts with %, then ignore it for now
    if(file.startsWith('%')) {
      console.log('ignoring', file);
      continue;
    }
    // if the file doesn't end with .png, ignore it'
    if(!file.endsWith('.png')) {
      console.log('ignoring', file);
      continue;
    }
    const dateDirName = await getDateDirName(filePath);
    const modelDirName = await getModelDirName(filePath);
    const destDir = path.join('./out', dateDirName, modelDirName);
    console.log({destDir});
  }
}
const main = async () => {
  await findImageFiles('./in/');

}

main();
