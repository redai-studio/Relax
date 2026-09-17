// Copyright (c) 2026 Relax Authors. All Rights Reserved.
const {release, reportError} = require('./common.js');

release(process.env.STATE_lease_directory)
    .then(() => console.log('Device reservation cleanup complete'))
    .catch(reportError);
