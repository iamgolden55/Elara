# Elara AI: Deployment Guide

This guide provides detailed instructions for deploying the Elara AI medical assistant in different environments, from local development to production. It covers containerization, scaling, security considerations, and monitoring.

## Deployment Options

Elara AI can be deployed in several ways:

1. **Local Development**: Run directly on your machine for development and testing
2. **Docker Containers**: Containerized deployment for consistency and isolation
3. **Cloud Deployment**: Hosted on cloud platforms for scalability and availability
4. **On-Premises**: Deployed within a healthcare organization's infrastructure

## Local Development Deployment

For local development and testing, follow these steps:

### Prerequisites

- Python 3.10+
- Node.js 16+ (for frontend)
- Git

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/elara-ai.git
cd elara-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model files (not included in repository)
# Place them in the models_files directory
python scripts/download_mistral.py  # If you have a script for this

# Run the backend server
cd backend
python main.py
```

The backend will be available at http://localhost:8000.

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at http://localhost:3000.

## Docker Deployment

Docker provides a consistent deployment environment and simplifies dependency management.

### Prerequisites

- Docker
- Docker Compose

### Building and Running with Docker Compose

```bash
# Clone the repository
git clone https://github.com/yourusername/elara-ai.git
cd elara-ai

# Build and start the containers
docker-compose up --build
```

This will start both the backend and frontend services. The backend will be available at http://localhost:8000 and the frontend at http://localhost:3000.

### Docker Compose Configuration

The `docker-compose.yml` file defines the services:

```yaml
version: "3.9"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./models_files:/app/models_files:ro  # Read-only mount of model files
    environment:
      - ELARA_DEBUG=false
      - ELARA_HOST=0.0.0.0
      - ELARA_PORT=8000
    restart: unless-stopped
    
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped
```

### Building Individual Services

You can also build and run services individually:

```bash
# Build and run backend only
docker-compose up --build backend

# Build and run frontend only
docker-compose up --build frontend
```

## Cloud Deployment

### Amazon Web Services (AWS)

#### Using Amazon ECS (Elastic Container Service)

1. **Create an ECR Repository**:
   ```bash
   aws ecr create-repository --repository-name elara-backend
   aws ecr create-repository --repository-name elara-frontend
   ```

2. **Build and Push Docker Images**:
   ```bash
   # Log in to ECR
   aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com
   
   # Build and tag images
   docker build -t your-account-id.dkr.ecr.your-region.amazonaws.com/elara-backend:latest -f Dockerfile.backend .
   docker build -t your-account-id.dkr.ecr.your-region.amazonaws.com/elara-frontend:latest -f Dockerfile.frontend .
   
   # Push images
   docker push your-account-id.dkr.ecr.your-region.amazonaws.com/elara-backend:latest
   docker push your-account-id.dkr.ecr.your-region.amazonaws.com/elara-frontend:latest
   ```

3. **Create ECS Task Definition**:
   Create a task definition that includes both containers.

4. **Create ECS Service**:
   Create a service that runs the task definition, with appropriate networking and load balancing.

#### Using AWS Elastic Beanstalk

1. **Create Dockerrun.aws.json**:
   ```json
   {
     "AWSEBDockerrunVersion": 2,
     "containerDefinitions": [
       {
         "name": "elara-backend",
         "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/elara-backend:latest",
         "essential": true,
         "memory": 4096,
         "portMappings": [
           {
             "hostPort": 8000,
             "containerPort": 8000
           }
         ]
       },
       {
         "name": "elara-frontend",
         "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/elara-frontend:latest",
         "essential": true,
         "memory": 512,
         "portMappings": [
           {
             "hostPort": 80,
             "containerPort": 80
           }
         ],
         "links": [
           "elara-backend"
         ]
       }
     ]
   }
   ```

2. **Deploy to Elastic Beanstalk**:
   ```bash
   eb init -p docker
   eb create elara-production
   ```

### Google Cloud Platform (GCP)

#### Using Google Kubernetes Engine (GKE)

1. **Push Images to Google Container Registry**:
   ```bash
   # Tag images
   docker tag elara-backend:latest gcr.io/your-project-id/elara-backend:latest
   docker tag elara-frontend:latest gcr.io/your-project-id/elara-frontend:latest
   
   # Push images
   docker push gcr.io/your-project-id/elara-backend:latest
   docker push gcr.io/your-project-id/elara-frontend:latest
   ```

2. **Create Kubernetes Deployment Files**:
   
   `backend-deployment.yaml`:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: elara-backend
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: elara-backend
     template:
       metadata:
         labels:
           app: elara-backend
       spec:
         containers:
         - name: elara-backend
           image: gcr.io/your-project-id/elara-backend:latest
           ports:
           - containerPort: 8000
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: elara-backend
   spec:
     selector:
       app: elara-backend
     ports:
     - port: 8000
       targetPort: 8000
     type: ClusterIP
   ```

   `frontend-deployment.yaml`:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: elara-frontend
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: elara-frontend
     template:
       metadata:
         labels:
           app: elara-frontend
       spec:
         containers:
         - name: elara-frontend
           image: gcr.io/your-project-id/elara-frontend:latest
           ports:
           - containerPort: 80
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: elara-frontend
   spec:
     selector:
       app: elara-frontend
     ports:
     - port: 80
       targetPort: 80
     type: LoadBalancer
   ```

3. **Deploy to GKE**:
   ```bash
   # Create a cluster
   gcloud container clusters create elara-cluster --num-nodes=2
   
   # Apply deployment files
   kubectl apply -f backend-deployment.yaml
   kubectl apply -f frontend-deployment.yaml
   ```

### Microsoft Azure

#### Using Azure Container Instances (ACI)

1. **Create a Resource Group**:
   ```bash
   az group create --name elara-group --location eastus
   ```

2. **Create a Container Registry**:
   ```bash
   az acr create --resource-group elara-group --name elararegistry --sku Basic
   ```

3. **Build and Push Images**:
   ```bash
   # Log in to registry
   az acr login --name elararegistry
   
   # Tag images
   docker tag elara-backend:latest elararegistry.azurecr.io/elara-backend:latest
   docker tag elara-frontend:latest elararegistry.azurecr.io/elara-frontend:latest
   
   # Push images
   docker push elararegistry.azurecr.io/elara-backend:latest
   docker push elararegistry.azurecr.io/elara-frontend:latest
   ```

4. **Deploy with Docker Compose to ACI**:
   Create a `docker-compose.yml` file and deploy using Azure CLI:
   ```bash
   az container app up --resource-group elara-group --name elara-app --compose-file docker-compose.yml
   ```

## On-Premises Deployment

For healthcare organizations that require on-premises deployment for security or compliance reasons:

### Using Docker Swarm

1. **Initialize Docker Swarm**:
   ```bash
   docker swarm init
   ```

2. **Deploy with Docker Stack**:
   ```bash
   docker stack deploy -c docker-compose.yml elara
   ```

### Using Kubernetes On-Premises

1. **Set Up Kubernetes Cluster** (using tools like kubeadm, k3s, or a managed Kubernetes solution)

2. **Deploy Using Kubernetes Manifests**:
   ```bash
   kubectl apply -f backend-deployment.yaml
   kubectl apply -f frontend-deployment.yaml
   ```

## Production Considerations

### High Availability

For production deployments, ensure high availability:

1. **Multiple Replicas**: Run multiple instances of each service
2. **Load Balancing**: Distribute traffic across instances
3. **Health Checks**: Monitor service health and restart failed instances
4. **Automated Scaling**: Scale based on load
5. **Geographical Distribution**: Deploy across multiple regions for global availability

### Scaling

Configure your deployment to scale appropriately:

1. **Horizontal Scaling**: Add more instances as load increases
2. **Vertical Scaling**: Increase resources (CPU, memory) for individual instances
3. **Auto-scaling**: Configure auto-scaling based on metrics like CPU usage, memory usage, or request count

Example Kubernetes HPA (Horizontal Pod Autoscaler):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: elara-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: elara-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Security

Secure your deployment:

1. **HTTPS**: Use TLS/SSL for all communications
2. **API Security**: Implement authentication and authorization
3. **Network Security**: Use network policies to control traffic
4. **Secrets Management**: Securely manage secrets using tools like Kubernetes Secrets, HashiCorp Vault, or AWS Secrets Manager
5. **Container Security**: Scan images for vulnerabilities
6. **Access Control**: Implement strict access controls
7. **Audit Logging**: Enable comprehensive audit logging

### Resource Requirements

Ensure your deployment environment meets these minimum requirements:

| Component | CPU | Memory | Storage | GPU (Optional) |
|-----------|-----|--------|---------|----------------|
| Backend   | 2-4 cores | 8-16 GB | 20 GB | 1 GPU (for larger models) |
| Frontend  | 1-2 cores | 2-4 GB  | 5 GB  | N/A |
| Database  | 2 cores   | 4 GB    | 20 GB | N/A |

For larger deployments or models, increase resources accordingly.

## Monitoring and Logging

### Monitoring

Set up monitoring for your deployment:

1. **Prometheus**: Collect metrics
2. **Grafana**: Visualize metrics
3. **Alerting**: Configure alerts for issues

Example Prometheus configuration:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'elara-backend'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['elara-backend:8000']

  - job_name: 'elara-frontend'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['elara-frontend:80']
```

### Logging

Implement comprehensive logging:

1. **Centralized Logging**: Use ELK Stack (Elasticsearch, Logstash, Kibana) or similar
2. **Log Rotation**: Configure log rotation to manage disk space
3. **Structured Logging**: Use structured logging format (JSON)

Example logging configuration for the backend:
```python
import logging
import json
from logging.handlers import RotatingFileHandler

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/elara.log',
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger
```

## CI/CD Pipeline

Implement a CI/CD pipeline for automated testing and deployment:

### GitHub Actions Example

Create a `.github/workflows/deploy.yml` file:

```yaml
name: Deploy Elara AI

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push backend
        uses: docker/build-push-action@v3
        with:
          context: .
          file: ./Dockerfile.backend
          push: true
          tags: yourusername/elara-backend:latest
      - name: Build and push frontend
        uses: docker/build-push-action@v3
        with:
          context: .
          file: ./Dockerfile.frontend
          push: true
          tags: yourusername/elara-frontend:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        # This step depends on your deployment environment
        # For example, deploy to Kubernetes:
        run: |
          echo ${{ secrets.KUBECONFIG }} > kubeconfig
          export KUBECONFIG=./kubeconfig
          kubectl rollout restart deployment/elara-backend
          kubectl rollout restart deployment/elara-frontend
```

## Healthcare-Specific Deployment Considerations

### HIPAA Compliance (US)

If deployed in a healthcare setting in the US, ensure HIPAA compliance:

1. **Business Associate Agreement (BAA)**: Ensure you have BAAs with all service providers
2. **Encryption**: Encrypt data at rest and in transit
3. **Access Controls**: Implement role-based access controls
4. **Audit Logging**: Maintain comprehensive audit logs
5. **Backup and Recovery**: Implement backup and disaster recovery procedures
6. **Risk Assessment**: Conduct regular risk assessments
7. **Policies and Procedures**: Develop and implement security policies and procedures

### GDPR Compliance (EU)

For deployments in the EU, ensure GDPR compliance:

1. **Data Minimization**: Collect and process only necessary data
2. **Consent**: Obtain and manage user consent
3. **User Rights**: Implement mechanisms for data access, rectification, and deletion
4. **Data Protection Impact Assessment**: Conduct assessments for high-risk processing
5. **Data Breach Notification**: Implement procedures for timely breach notification
6. **Privacy by Design**: Implement privacy by design principles

### Other Regional Regulations

Be aware of and comply with other regional regulations:

1. **PIPEDA** (Canada)
2. **POPIA** (South Africa)
3. **PDPA** (Singapore)
4. **Regional Health Data Regulations**: Many countries have specific healthcare data regulations

## Custom Deployment Scenarios

### Embedded Deployment

For embedding Elara AI into existing healthcare systems:

1. **API Integration**: Provide RESTful APIs for integration
2. **SDK Development**: Develop SDKs for common programming languages
3. **Webhook Support**: Implement webhooks for event-driven integration
4. **Single Sign-On**: Support SSO for seamless user experience

### Air-Gapped Deployment

For high-security environments without internet access:

1. **Offline Installation**: Provide offline installation packages
2. **Manual Updates**: Support manual updates for models and software
3. **Local Knowledge Base**: Pre-build and package the vector database
4. **Offline Documentation**: Include comprehensive documentation in the deployment

## Performance Optimization

Optimize performance for your deployment:

1. **Model Quantization**: Use quantized models (INT8, INT4) for faster inference
2. **Caching**: Implement response caching for common queries
3. **Load Balancing**: Distribute traffic across multiple instances
4. **Asynchronous Processing**: Use asynchronous processing for non-critical tasks
5. **Database Optimization**: Optimize database queries and indexing
6. **CDN**: Use a Content Delivery Network for static assets

## Troubleshooting

Common deployment issues and solutions:

### Backend Service Not Starting

**Symptoms**:
- Container exits immediately
- Error in logs: "Failed to load model"

**Solutions**:
1. Check if model files are correctly mounted
2. Ensure sufficient memory is allocated
3. Verify environment variables are correctly set
4. Check log files for specific errors

### Frontend Cannot Connect to Backend

**Symptoms**:
- Frontend displays "Cannot connect to server"
- Network errors in browser console

**Solutions**:
1. Verify backend service is running
2. Check network configuration (ports, DNS)
3. Ensure CORS is correctly configured
4. Check firewall rules

### Model Loading Errors

**Symptoms**:
- Error: "Failed to load model weights"
- Error: "CUDA out of memory"

**Solutions**:
1. Verify model files are correctly placed
2. Ensure sufficient GPU memory (or use CPU fallback)
3. Try quantized models for lower memory usage
4. Check if CUDA/GPU drivers are correctly installed

### Slow Response Times

**Symptoms**:
- API requests take too long to complete
- Frontend seems unresponsive

**Solutions**:
1. Use a more efficient model or quantize the current one
2. Implement caching for common queries
3. Scale horizontally by adding more instances
4. Optimize database queries
5. Use streaming responses for long-running operations

## Conclusion

Deploying Elara AI requires careful planning and consideration of various factors, including infrastructure, security, compliance, and performance. This guide provides a comprehensive overview of deployment options and best practices to help you successfully deploy Elara AI in different environments.

Remember that deployment is not a one-time activity but an ongoing process. Regularly update your deployment with the latest security patches, model improvements, and feature enhancements to ensure the best user experience and maintain compliance with evolving regulations.

By following the guidelines in this document, you can create a robust, scalable, and secure deployment of Elara AI that meets the needs of your users and organization.
