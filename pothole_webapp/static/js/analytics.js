// Analytics Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadAnalyticsData();
});

// Chart configuration with dark theme support
const chartConfig = {
    fontColor: '#f1f5f9',
    gridColor: 'rgba(100, 116, 139, 0.3)',
    borderColor: 'rgba(100, 116, 139, 0.5)',
    backgroundColor: 'rgba(30, 41, 59, 0.8)'
};

function initializeCharts() {
    // Set Chart.js default colors for dark theme
    Chart.defaults.color = chartConfig.fontColor;
    Chart.defaults.borderColor = chartConfig.borderColor;

    // Accident Trends Chart
    const accidentCtx = document.getElementById('accidentChart').getContext('2d');
    new Chart(accidentCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: 'Pothole Accidents',
                data: [185, 192, 210, 198, 205, 235, 310, 285, 260, 240, 220, 195],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#ef4444',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Monthly Pothole Accident Trends',
                    color: chartConfig.fontColor,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor
                    },
                    title: {
                        display: true,
                        text: 'Number of Accidents',
                        color: chartConfig.fontColor,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor
                    }
                }
            }
        }
    });

    // Severity Distribution Chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    new Chart(severityCtx, {
        type: 'doughnut',
        data: {
            labels: ['Minor Damage', 'Vehicle Repair', 'Injuries', 'Fatalities'],
            datasets: [{
                data: [45, 30, 20, 5],
                backgroundColor: [
                    '#10b981', // Green
                    '#f59e0b', // Amber
                    '#f97316', // Orange
                    '#ef4444'  // Red
                ],
                borderColor: '#1e293b',
                borderWidth: 3,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Accident Severity Distribution',
                    color: chartConfig.fontColor,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 12
                        },
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            },
            cutout: '50%'
        }
    });

    // Cost Breakdown Chart
    const costCtx = document.getElementById('costChart').getContext('2d');
    new Chart(costCtx, {
        type: 'bar',
        data: {
            labels: ['Vehicle Repair', 'Medical Costs', 'Traffic Delays', 'Manual Inspection', 'Legal Claims'],
            datasets: [{
                label: 'Cost in Crores (₹)',
                data: [2.1, 0.8, 0.9, 0.6, 0.3],
                backgroundColor: [
                    '#3b82f6', // Blue
                    '#ef4444', // Red
                    '#f59e0b', // Amber
                    '#10b981', // Green
                    '#8b5cf6'  // Purple
                ],
                borderColor: '#1e293b',
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Annual Pothole-Related Costs',
                    color: chartConfig.fontColor,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor,
                        callback: function(value) {
                            return '₹' + value + 'Cr';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Cost (₹ Crores)',
                        color: chartConfig.fontColor,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });

    // Trend Chart - Traditional vs AI Costs
    const trendCtx = document.getElementById('trendChart').getContext('2d');
    new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: ['2019', '2020', '2021', '2022', '2023', '2024'],
            datasets: [
                {
                    label: 'Traditional Methods Cost',
                    data: [3.2, 3.5, 3.8, 4.0, 4.2, 4.5],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#ef4444',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                },
                {
                    label: 'AI System Cost',
                    data: [2.8, 2.5, 2.2, 1.9, 1.7, 1.5],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Cost Comparison: Traditional vs AI Systems',
                    color: chartConfig.fontColor,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor,
                        callback: function(value) {
                            return '₹' + value + 'Cr';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Cost (₹ Crores)',
                        color: chartConfig.fontColor,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor
                    }
                }
            }
        }
    });

    // Regional Chart - Accidents by City
    const regionalCtx = document.getElementById('regionalChart').getContext('2d');
    new Chart(regionalCtx, {
        type: 'bar',
        data: {
            labels: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune'],
            datasets: [{
                label: 'Pothole Accidents',
                data: [428, 392, 315, 287, 234, 198, 176],
                backgroundColor: [
                    '#ef4444', // Red
                    '#f97316', // Orange
                    '#f59e0b', // Amber
                    '#eab308', // Yellow
                    '#84cc16', // Lime
                    '#22c55e', // Green
                    '#10b981'  // Emerald
                ],
                borderColor: '#1e293b',
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Pothole Accidents by City (Annual)',
                    color: chartConfig.fontColor,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor
                    },
                    title: {
                        display: true,
                        text: 'Number of Accidents',
                        color: chartConfig.fontColor,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    grid: {
                        color: chartConfig.gridColor
                    },
                    ticks: {
                        color: chartConfig.fontColor,
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

function loadAnalyticsData() {
    console.log('Analytics data loaded successfully!');
    
    // Add some interactive effects
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Add resize handler for better responsiveness
window.addEventListener('resize', function() {
    // Charts will automatically resize due to responsive: true
    console.log('Window resized - charts updated');
});