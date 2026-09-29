#!/bin/bash
# Eval2 body: zero-shot fusion kinds. Sets STUDY_EVAL_SUITE=eval2 then runs shared eval driver.

export STUDY_EVAL_SUITE=eval2
_STUDY_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=study_eval_job.sh
source "${_STUDY_JOB_DIR}/study_eval_job.sh"
