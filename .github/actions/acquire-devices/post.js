// Copyright (c) 2026 Relax Authors. All Rights Reserved.
import {release, reportError} from './common.js';

release(process.env.STATE_lease_directory)
    .then(() => console.log('Device reservation cleanup complete'))
    .catch(reportError);
