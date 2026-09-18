// Copyright (c) 2026 Relax Authors. All Rights Reserved.
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';

import {release, reportError, requestRelease, writeCommand} from './common.js';

function input(name, fallback) {
  return (process.env[`INPUT_${name.toUpperCase()}`] ?? fallback).trim();
}

function integer(name, fallback, minimum) {
  const value = input(name, fallback);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) ||
      Number(value) < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return Number(value);
}

async function main() {
  if (process.env.RUNNER_OS !== 'Linux')
    throw new Error('acquire-devices requires a Linux runner');
  const config = {
    backend : input('backend', 'nvidia'),
    count : integer('count', '1', 1),
    devices : input('devices', ''),
    timeout : integer('timeout', '1800', 0),
    lock_dir : input('lock-dir', '/tmp/acquire-devices'),
    parent_pid : process.pid,
  };
  if (config.backend !== 'nvidia')
    throw new Error(
        `Unsupported backend: ${config.backend}; supported: nvidia`);
  if (!path.isAbsolute(config.lock_dir))
    throw new Error('lock-dir must be an absolute host path');
  for (const name of ['GITHUB_STATE', 'GITHUB_OUTPUT', 'GITHUB_ENV']) {
    if (!process.env[name])
      throw new Error(`Missing ${name}`);
  }

  const directory = fs.mkdtempSync(
      path.join(process.env.RUNNER_TEMP || os.tmpdir(), 'acquire-devices-'));
  const logPath = path.join(directory, 'holder.log');
  const resultPath = path.join(directory, 'result.json');
  // Register cleanup before starting any process that could hold locks.
  writeCommand(process.env.GITHUB_STATE, 'lease_directory', directory);
  const handlers = new Map();
  for (const [signal, code] of [[ 'SIGINT', 130 ], [ 'SIGTERM', 143 ]]) {
    const handler = () => {
      requestRelease(directory);
      process.exit(code);
    };
    handlers.set(signal, handler);
    process.once(signal, handler);
  }

  try {
    fs.writeFileSync(path.join(directory, 'config.json'),
                     JSON.stringify(config));
    const log = fs.openSync(logPath, 'a');
    let child;
    try {
      child = spawn('python3',
                    [ path.join(import.meta.dirname, 'lease.py'), directory ], {
                      detached : true,
                      stdio : [ 'ignore', log, log ],
                      // Preserve RUNNER_TRACKING_ID for the runner's final
                      // orphan cleanup.
                      env : process.env,
                    });
    } finally {
      fs.closeSync(log);
    }
    await new Promise((resolve, reject) => {
      child.once('spawn', resolve);
      child.once('error', reject);
    });
    fs.writeFileSync(path.join(directory, 'pid'), String(child.pid));
    console.log(`Waiting for ${config.count} ${
        config.backend} device(s), timeout ${config.timeout}s`);
    let logOffset = 0;
    while (true) {
      const finished = fs.existsSync(resultPath);
      const log = fs.readFileSync(logPath, 'utf8');
      process.stdout.write(log.slice(logOffset));
      logOffset = log.length;
      if (finished)
        break;
      if (child.exitCode !== null || child.signalCode !== null) {
        throw new Error('Device holder exited before allocation');
      }
      await sleep(50);
    }
    const result = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
    if (result.error)
      throw new Error(result.error);
    writeCommand(process.env.GITHUB_OUTPUT, 'devices',
                 result.devices.join(','));
    writeCommand(process.env.GITHUB_OUTPUT, 'count',
                 String(result.devices.length));
    writeCommand(process.env.GITHUB_ENV, result.visibility_env,
                 result.devices.join(','));
    fs.writeFileSync(path.join(directory, 'accepted'), '');
    child.unref();
    console.log(
        `Reserved ${config.backend} devices: ${result.devices.join(',')}`);
  } catch (error) {
    await release(directory);
    throw error;
  } finally {
    for (const [signal, handler] of handlers)
      process.removeListener(signal, handler);
  }
}

main().catch(reportError);
