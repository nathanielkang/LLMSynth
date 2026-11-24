#!/usr/bin/env bash
# Reminder to stop GPU VM after experiments (avoid idle charges).
: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project id}"
: "${GCP_INSTANCE:?Set GCP_INSTANCE to your VM name}"
PROJECT="$GCP_PROJECT"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="$GCP_INSTANCE"

cat <<MSG
After downloading results, stop the VM:

  gcloud compute instances stop ${INSTANCE} --project ${PROJECT} --zone ${ZONE}

Optional delete (irreversible):

  gcloud compute instances delete ${INSTANCE} --project ${PROJECT} --zone ${ZONE}
MSG
