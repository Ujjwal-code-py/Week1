# 🛣️ Pothole Inspector Pro

AI-Powered Road Infrastructure Analysis & Cost Estimation Platform

Transforming road maintenance with computer vision and deep learning


![Pi7_GIF_CMP](https://github.com/user-attachments/assets/1656ccd5-28c1-46c1-8cbb-9d932f1b2940)


</div>

## Dataset
- https://www.kaggle.com/datasets/anugrahakbar/potholes-detection-for-yolov4

📖 Table of Contents
- Overview

- Features

- Quick Start

- Installation

- Usage Guide

- API Documentation

- Developer Guide

- Deployment

- Support
  
## Overview
### Pothole Inspector Pro is a comprehensive web application that leverages artificial intelligence to automatically detect potholes in road images and videos. It provides detailed analysis including dimension measurement, repair cost estimation, and professional reporting for road maintenance teams, government agencies, and infrastructure companies.

##  What Problem We Solve
- Manual Inspection Costs: Reduces time and cost of manual road surveys
- Data-Driven Decisions: Provides accurate measurements and cost estimates
- Preventive Maintenance: Identifies high-risk areas before accidents occur
- Documentation: Generates professional reports for audit and planning

## ✨ Features
<img width="980" height="454" alt="image" src="https://github.com/user-attachments/assets/af48dbac-82e2-4f3d-95b8-1322bd7be449" />

## 📊 Advanced Analytics
- Interactive Dashboard with Chart.js visualizations
- Regional Risk Analysis identifying high-priority zones
- Historical Trend Analysis for preventive maintenance
- Economic Impact Assessment with cost-benefit analysis

##  User Experience
- Drag & Drop Interface for easy file upload
- GPS Auto-Location with reverse geocoding
- Real-time Progress Tracking during processing
- Responsive Design works on all devices

##  Quick Start
- Upload a road image or video
- Enter location and cost parameters
- Get instant analysis with cost breakdown
- Download professional PDF report

## Installation
### Clone & setup
- git clone https://github.com/your-username/pothole-detection.git
- cd pothole_detection
- pip install -r requirements.txt

### Configure environment
- cp .env.example .env
### Edit .env with your Cloudinary credentials

### Run application
- cd pothole_webapp
- python app.py

### Prerequisites
- Python 3.13.2 (Required)

- pip package manager

- Cloudinary account (free tier available)

## 📱 Usage Guide
1. Upload Media
- Drag & drop or click to upload images/videos
- Supported formats: JPG, PNG, MP4, AVI, MOV
- Max size: 16MB

2. Location Information
- Click 📍 Get Location for automatic GPS detection
Or manually enter: Road name, city, coordinates
Additional notes for context

3. Cost Parameters
- Material Cost: ₹ per liter (40-55 range)

- Labor Cost: ₹ per hour per worker (300-400 range)

- Team Size: Number of workers (typically 2-4)

- Overhead: Percentage for admin costs (10-20%)

4. View Results
- Pothole Count: Total detected potholes

- Dimensions: Width, depth, volume for each pothole

- Cost Breakdown: Detailed repair cost estimation

- Annotated Image: Visual result with bounding boxes

5. Export Reports
Click Download PDF Report for professional documentation
Report includes all analysis data and cost breakdowns

## 📊 Analytics Dashboard
- Visit /analytics for comprehensive insights
- View accident trends and economic impact
- Identify high-risk zones
- Track historical data

## 🔌 API Documentation
<img width="1001" height="360" alt="image" src="https://github.com/user-attachments/assets/870825bd-3f08-49e0-b12d-a64985c9c061" />

### Location Services API
The application uses OpenStreetMap Nominatim API for reverse geocoding:
<img width="967" height="330" alt="image" src="https://github.com/user-attachments/assets/bb4d47bd-fabb-4bcf-9082-c5349f341daf" />

- Free and open-source geocoding service
- No API key required
- 
## 🛠️ Developer Guide

### Project Structure
<img width="892" height="545" alt="image" src="https://github.com/user-attachments/assets/73e2e49d-a1fe-4c7a-9564-c6b4ece7d4c3" />

## 🙏 Acknowledgments
YOLOv8 by Ultralytics for object detection

Cloudinary for reliable cloud storage

OpenCV for computer vision processing

Chart.js for analytics visualization

ReportLab for PDF generation
<div align="center">
Built with ❤️ using Python 3.13.2

Making roads safer, one pothole at a time

</div>
## Referance:
https://www.ijmrset.com/upload/30_Pothole.pdf
