#!/usr/bin/env bash
# Reminder to stop GPU VM after experiments (avoid idle charges).
PROJECT="${GCP_PROJECT:-knu-fog-experiments}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-llmsynth-gpu}"

cat <<MSG
After downloading results, stop the VM:

  gcloud compute instances stop ${INSTANCE} --project ${PROJECT} --zone ${ZONE}

Optional delete (irreversible):

  gcloud compute instances delete ${INSTANCE} --project ${PROJECT} --zone ${ZONE}
MSG
