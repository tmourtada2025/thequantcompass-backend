#!/bin/bash

# Google Cloud Project Configuration
PROJECT_ID="awesome-sphere-449619-e4"
SERVICE_NAME="quantcompass-backend"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying QuantCompass Backend to Google Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Set the project
echo "📋 Setting Google Cloud project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Build the Docker image
echo "🏗️ Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 1 \
    --max-instances 10 \
    --port 8080 \
    --set-env-vars "PORT=8080" \
    --set-env-vars "FLASK_ENV=production" \
    --timeout 300

# Get the service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo "🌐 Service URL: ${SERVICE_URL}"

# Test the health endpoint
echo "🔍 Testing health endpoint..."
curl -f "${SERVICE_URL}/api/health" || echo "❌ Health check failed"

echo ""
echo "🎉 QuantCompass Backend deployed successfully!"
echo "📊 Dashboard: https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics?project=${PROJECT_ID}"
echo "🔗 API Base URL: ${SERVICE_URL}"
echo ""
echo "📋 Next steps:"
echo "1. Set up environment variables in Cloud Run console"
echo "2. Configure Polygon.io API key"
echo "3. Set up email/notification credentials"
echo "4. Test the API endpoints"
