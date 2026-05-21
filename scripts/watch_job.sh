#!/bin/bash
# Run on a MetaCentrum frontend (e.g. tarkil). Tails stdout/stderr of a running job.
# Source: https://docs.metacentrum.cz/en/docs/computing/jobs/job-tracking
# Usage: ./watch_job.sh <jobid_number> [OU|ER]
#   ./watch_job.sh 20422010          # follows .OU
#   ./watch_job.sh 20422010 ER       # follows .ER

set -e

JOBID_NUM="${1:?Usage: $0 <jobid_number> [OU|ER]}"
STREAM="${2:-OU}"
FULL_JOBID="${JOBID_NUM}.pbs-m1.metacentrum.cz"

EXEC_LINE=$(qstat -f "$FULL_JOBID" | grep exec_host2)
# parses lines like:  exec_host2 = zenon41.cerit-sc.cz:15002/12
EXEC_HOST=$(echo "$EXEC_LINE" | sed -E 's/.*=[[:space:]]*([^:]+):.*/\1/')

if [ -z "$EXEC_HOST" ]; then
    echo "Could not determine exec host. Raw line: $EXEC_LINE"
    exit 1
fi

SPOOL_FILE="/var/spool/pbs/spool/${FULL_JOBID}.${STREAM}"
echo "Job is on $EXEC_HOST — tailing $SPOOL_FILE (Ctrl-C to stop)"
echo

ssh -t "$EXEC_HOST" "tail -f $SPOOL_FILE"
