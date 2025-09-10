# Bank Statement Processing Application - AWS EC2 Deployment Guide

## Overview
This document provides a comprehensive guide for deploying a bank statement processing application on AWS EC2 using Docker Compose. The application consists of a Django backend with OCR capabilities and a React frontend served by Nginx.

## Application Architecture
- **Backend**: Django REST API with OCR processing using Hugging Face models
- **Frontend**: React TypeScript application with Vite build system
- **Database**: SQLite (can be upgraded to PostgreSQL for production)
- **Web Server**: Nginx for frontend serving
- **Containerization**: Docker and Docker Compose

## Prerequisites
- AWS Account with EC2 access
- Basic knowledge of Linux commands
- Git installed locally
- Docker and Docker Compose installed on EC2 instance

## Step 1: AWS EC2 Instance Setup

### 1.1 Create EC2 Instance
1. Log into AWS Console
2. Navigate to EC2 Dashboard
3. Click "Launch Instance"
4. **Instance Configuration**:
   - **Name**: `bank-statement-app`
   - **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance Type**: `t3.xlarge` (4 vCPUs, 16 GB RAM)
   - **Key Pair**: Create or select existing key pair
   - **Security Group**: Create new security group with following rules:
     - SSH (22): Your IP
     - HTTP (80): 0.0.0.0/0
     - Custom TCP (3000): 0.0.0.0/0
     - Custom TCP (8000): 0.0.0.0/0
   - **Storage**: 30 GB gp3 (minimum)

### 1.2 Connect to Instance
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

## Step 2: System Dependencies Installation

### 2.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

### 2.3 Install Docker Compose
```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2.4 Install Git
```bash
sudo apt install git -y
```

### 2.5 Logout and Login
```bash
exit
# SSH back in to apply docker group changes
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

## Step 3: Application Deployment

### 3.1 Clone Repository
```bash
# Create application directory
mkdir -p ~/bank-statement-app
cd ~/bank-statement-app

# Clone your repository (replace with your actual repository URL)
git clone https://github.com/yourusername/your-repo-name.git .
```

### 3.2 Create Environment Configuration
Create a `.env` file in the project root:

```bash
nano .env
```

Add the following content (replace with your actual values):

```env
# Hugging Face API Configuration
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
HUGGINGFACE_MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct

# Django Configuration
SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,your-ec2-private-ip,your-ec2-public-ip
CORS_ALLOW_ALL_ORIGINS=True
```

**Important**: 
- Replace `your_huggingface_token_here` with your actual Hugging Face API token
- Replace `your_django_secret_key_here` with a secure Django secret key
- Replace `your-ec2-private-ip` and `your-ec2-public-ip` with your actual EC2 IP addresses

### 3.3 Create Database File
```bash
# Create SQLite database file with proper permissions
touch db.sqlite3
chmod 666 db.sqlite3
```

### 3.4 Build and Start Application
```bash
# Build and start containers
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3.5 Run Database Migrations
```bash
# Run Django migrations
docker-compose exec backend python manage.py migrate
```

## Step 4: Verification and Testing

### 4.1 Test Backend API
```bash
# Test backend health
curl http://localhost:8000/

# Test API endpoints
curl -X GET http://localhost:8000/api/pdf/

# Test admin interface
curl http://localhost:8000/admin/
```

### 4.2 Test Frontend
```bash
# Test frontend
curl http://localhost:3000/
```

### 4.3 Test External Access
Replace `your-ec2-public-ip` with your actual public IP:

```bash
# Test backend from external IP
curl http://your-ec2-public-ip:8000/

# Test frontend from external IP
curl http://your-ec2-public-ip:3000/
```

## Step 5: Security Configuration

### 5.1 Update .env for Production
For production deployment, update your `.env` file:

```env
# Production Configuration
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,your-ec2-public-ip
CORS_ALLOW_ALL_ORIGINS=False
CORS_ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

### 5.2 Generate Secure Secret Key
```bash
# Generate a new Django secret key
python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

## Step 6: Troubleshooting Common Issues

### 6.1 Database Permission Errors
If you encounter `sqlite3.OperationalError: unable to open database file`:

```bash
# Stop containers
docker-compose down

# Remove problematic database file
sudo rm -rf db.sqlite3

# Create new database file with proper permissions
touch db.sqlite3
chmod 666 db.sqlite3

# Restart containers
docker-compose up -d

# Run migrations
sleep 10
docker-compose exec backend python manage.py migrate
```

### 6.2 DisallowedHost Error
If you get `DisallowedHost` error when accessing via public IP:

1. Update `.env` file to include your public IP in `ALLOWED_HOSTS`
2. Restart containers:
```bash
docker-compose down
docker-compose up -d
```

### 6.3 Container Build Issues
If containers fail to build:

```bash
# Clean up Docker cache
docker system prune -a

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

### 6.4 Port Access Issues
Ensure your security group allows traffic on ports 80, 3000, and 8000:

1. Go to EC2 Dashboard → Security Groups
2. Select your instance's security group
3. Edit inbound rules to include:
   - Type: HTTP, Port: 80, Source: 0.0.0.0/0
   - Type: Custom TCP, Port: 3000, Source: 0.0.0.0/0
   - Type: Custom TCP, Port: 8000, Source: 0.0.0.0/0

## Step 7: Application Management

### 7.1 View Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 7.2 Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

### 7.3 Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 7.4 Backup Database
```bash
# Create backup
cp db.sqlite3 db_backup_$(date +%Y%m%d_%H%M%S).sqlite3
```

## Step 8: Monitoring and Maintenance

### 8.1 System Resources
```bash
# Check system resources
htop
df -h
free -h
```

### 8.2 Container Resources
```bash
# Check container resource usage
docker stats
```

### 8.3 Application Health
```bash
# Check if services are running
docker-compose ps

# Test application endpoints
curl -f http://localhost:8000/ || echo "Backend down"
curl -f http://localhost:3000/ || echo "Frontend down"
```

## Step 9: Production Considerations

### 9.1 Database Upgrade
For production, consider upgrading to PostgreSQL:

1. Update `requirements.txt` to include `psycopg2-binary`
2. Update `docker-compose.yml` to include PostgreSQL service
3. Update Django settings to use PostgreSQL

### 9.2 SSL/HTTPS Setup
1. Use AWS Certificate Manager for SSL certificates
2. Set up Application Load Balancer
3. Configure HTTPS redirects

### 9.3 Domain Configuration
1. Point your domain to EC2 public IP
2. Update `ALLOWED_HOSTS` in `.env`
3. Configure DNS records

### 9.4 Monitoring Setup
1. Set up CloudWatch monitoring
2. Configure log aggregation
3. Set up alerts for service failures

## Step 10: Cleanup Commands

### 10.1 Stop Application
```bash
docker-compose down
```

### 10.2 Remove Containers and Images
```bash
docker-compose down --rmi all
docker system prune -a
```

### 10.3 Remove Application Directory
```bash
cd ~
rm -rf bank-statement-app
```

## Application URLs

After successful deployment, your application will be accessible at:

- **Frontend**: `http://your-ec2-public-ip:3000`
- **Backend API**: `http://your-ec2-public-ip:8000`
- **Admin Interface**: `http://your-ec2-public-ip:8000/admin/`

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Check application logs for errors
2. **Monthly**: Update system packages and Docker images
3. **Quarterly**: Review and rotate API keys and secrets

### Monitoring Checklist
- [ ] All containers are running (`docker-compose ps`)
- [ ] Backend API responds (`curl http://localhost:8000/`)
- [ ] Frontend loads (`curl http://localhost:3000/`)
- [ ] Database migrations are up to date
- [ ] No critical errors in logs
- [ ] System resources are within limits

## Conclusion

This deployment guide provides a complete walkthrough for deploying the bank statement processing application on AWS EC2. The application is now containerized, scalable, and ready for production use with proper security configurations.

For additional support or questions, refer to the application documentation or contact the development team.
