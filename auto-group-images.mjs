#!/usr/bin/env node
import * as fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import * as imageHashModule from 'image-hash';
import { parseArgs } from 'util';

const { imageHash } = imageHashModule;

// Find all PNGs in a directory asynchronously
const findPngFiles = async (dirPath) => {
  const files = await fs.readdir(dirPath, { withFileTypes: true });
  return files
    .filter(file => file.isFile() && path.extname(file.name).toLowerCase() === '.png')
    .map(file => path.join(dirPath, file.name));
};

// Preprocess the image using sharp and calculate its perceptual hash asynchronously
const calculateImageHash = async (imagePath) => {
  imagePath = path.resolve(imagePath);
  console.log(`Processing image: ${imagePath}`);

  try {
    const hash = await new Promise((resolve, reject) => {
      imageHash(imagePath, 8, 'hex', (err, data) => {
        if (err) {
          console.error(`Error in imageHash for ${imagePath}:`, err);
          reject(err);
        } else {
          resolve(data);
        }
      });
    });

    console.log(`Hash for image ${imagePath}: ${hash}`);
    return { imagePath, hash };
  } catch (error) {
    console.error(`Error processing image ${imagePath}:`, error);
    return null;
  }
};

// Copy each file to the appropriate output folder based on sorted group index
const copyGroupedImages = async (groups, outputDir) => {
  await Promise.all(
    groups.map(async (group, sortedGroupIndex) => {
      const groupDir = path.join(outputDir, (sortedGroupIndex + 1).toString()); // Create a folder based on sorted index
      await fs.mkdir(groupDir, { recursive: true }); // Ensure the group folder exists

      await Promise.all(
        group.map(async (image) => {
          const destPath = path.join(groupDir, path.basename(image.imagePath)); // Destination path for the image
          await fs.copyFile(image.imagePath, destPath); // Copy the image to the destination
          console.log(`Copied ${image.imagePath} to ${destPath}`);
        })
      );
    })
  );
};

// Compare hashes and group similar images asynchronously
const groupImagesBySimilarity = async (hashedImages) => {
  const groups = [];
  const tolerance = 10; // Hash difference tolerance for grouping similar images

  await Promise.all(
    hashedImages.map(async (image) => {
      let addedToGroup = false;

      // Try to find an existing group for the image
      for (let group of groups) {
        const groupImageHash = group[0].hash;
        const hashDifference = getHammingDistance(groupImageHash, image.hash);

        if (hashDifference <= tolerance) {
          group.push(image);
          addedToGroup = true;
          break;
        }
      }

      // If no suitable group was found, create a new group
      if (!addedToGroup) {
        groups.push([image]);
      }
    })
  );

  return groups;
};

// Sort groups by the number of members in descending order
const sortGroupsBySize = (groups) => {
  return groups.sort((a, b) => b.length - a.length); // Sort by group size (largest first)
};

// Calculate the Hamming distance between two hex strings
const getHammingDistance = (hash1, hash2) => {
  let distance = 0;
  for (let i = 0; i < hash1.length; i++) {
    if (hash1[i] !== hash2[i]) {
      distance++;
    }
  }
  return distance;
};

// Main function to group images by similarity, copy them to sorted group folders, and print stride length
const main = async () => {
  const {
    values: { inputDir, outputDir },
  } = parseArgs({
    options: {
      inputDir: { type: 'string', short: 'i' },
      outputDir: { type: 'string', short: 'o' },
    },
  });

  if (!inputDir || !outputDir) {
    console.error('You must provide both --inputDir (-i) and --outputDir (-o).');
    process.exit(1);
  }

  // Find all PNG files in the input directory
  const pngFiles = await findPngFiles(inputDir);
  console.log(`Found ${pngFiles.length} PNG files.`);

  if (pngFiles.length === 0) {
    console.error('No PNG files found in the specified directory.');
    process.exit(1);
  }

  // Calculate hashes for all images
  console.log('Calculating hashes...');
  const hashedImages = (await Promise.all(pngFiles.map(calculateImageHash))).filter(Boolean);

  if (hashedImages.length === 0) {
    console.error('Failed to calculate hashes for the images.');
    process.exit(1);
  }

  // Group images by similarity
  const imageGroups = await groupImagesBySimilarity(hashedImages);

  // Sort groups by their number of members in descending order
  const sortedGroups = sortGroupsBySize(imageGroups);

  // The number of groups can be used as the stride length
  const strideLength = sortedGroups.length;

  console.log(`Stride length (number of groups): ${strideLength}`);

  // Copy grouped images to the output directory
  await copyGroupedImages(sortedGroups, outputDir);

  console.log('Images have been successfully copied to sorted group folders.');
};

main();
