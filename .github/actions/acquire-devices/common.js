// Copyright (c) 2026 Relax Authors. All Rights Reserved.
import fs from 'node:fs';
import path from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';

export function writeCommand(file, name, value) {
  if (!file)
    throw new Error(`Missing GitHub command file for ${name}`);
  // All exported values are validated single-line IDs, integers, or paths.
  if (/[\r\n]/.test(value))
    throw new Error(`Invalid multiline value for ${name}`);
  fs.appendFileSync(file, `${name}=${value}\n`);
}

export function requestRelease(directory) {
  if (fs.existsSync(directory))
    fs.writeFileSync(path.join(directory, 'release'), '');
}

export async function release(directory) {
  if (!directory || !fs.existsSync(directory))
    return;
  requestRelease(directory);
  const pidFile = path.join(directory, 'pid');
  const pid =
      fs.existsSync(pidFile) ? Number(fs.readFileSync(pidFile, 'utf8')) : null;
  const deadline = Date.now() + 15000;
  while (pid && !fs.existsSync(path.join(directory, 'released'))) {
    try {
      process.kill(pid, 0);
    } catch (error) {
      if (error.code === 'ESRCH')
        break;
      throw error;
    }
    if (Date.now() >= deadline) {
      throw new Error(`Device holder did not stop; inspect ${
          directory}. Shared lock files must not be deleted.`);
    }
    await sleep(50);
  }
  // Only the private control directory is removed, never the shared lock files.
  fs.rmSync(directory, {recursive : true, force : true});
}

export function reportError(error) {
  const message = String(error.message || error)
                      .replaceAll('%', '%25')
                      .replaceAll('\r', '%0D')
                      .replaceAll('\n', '%0A');
  console.error(`::error::${message}`);
  process.exitCode = 1;
}
