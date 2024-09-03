#!/usr/bin/env node
import * as fs from 'fs/promises';
import {exec as execCallBack} from 'child_process';
import {promisify} from 'util';
import path from 'path';
const exec = promisify(execCallBack);

const getDateDirName = async (imagePath) => {
  // get file modified date
  const {mtime} = await fs.stat(imagePath);
  return new Date(mtime).toISOString();
}

const getModelDirName = async (imagePath) => {
  // exec exiftool -p prompt imagePath
  const {stdout} = await exec(`exiftool -b '-prompt' ${imagePath}`);
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
    // await getModelDirName('./in/rave/1/3_00001_.png');
    // await getDateDirName('./in/rave/1/3_00001_.png');
    console.log("is file", filePath);
  }
}
const main = async () => {
  await findImageFiles('./in/');

}

main();
