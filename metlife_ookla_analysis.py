#!/usr/bin/env python3
"""
MetLife Stadium Ookla Q3 2025 Mobile Performance Analysis

This script downloads Ookla Q3 2025 mobile performance data, filters for
samples near MetLife Stadium, and creates an interactive Mapbox visualization.

MetLife Stadium Coordinates: 40.813778, -74.074310

Requirements:
    pip install duckdb pandas plotly shapely pyproj

Usage:
    python metlife_ookla_analysis.py

Author: Generated for syedazharmbnr1/Flowise
"""

import os
import sys
import subprocess

# Install required packages if not present
def install_packages():
    packages = ['duckdb', 'pandas', 'plotly', 'shapely', 'pyproj']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

install_packages()

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely import wkb
from shapely.geometry import Point, Polygon, box
from pyproj import Transformer
import json
import struct

# Configuration
MAPBOX_TOKEN = "pk.eyJ1IjoiYXpoYXJ6NHUiLCJhIjoiY2pkaHFtbHAxMGV1cDJxbzI0cjFlcWt4eiJ9.D-0A_N0JhPBfOm-CeFZtMQ"

# MetLife Stadium coordinates
METLIFE_LAT = 40.813778
METLIFE_LON = -74.074310

# Stadium approximate boundaries (roughly 300m radius around center)
SEARCH_RADIUS_KM = 0.5  # 500 meters search radius

# S3 URL for Ookla Q3 2025 mobile data
OOKLA_S3_PATH = "s3://ookla-open-data/parquet/performance/type=mobile/year=2025/quarter=3/2025-07-01_performance_mobile_tiles.parquet"
OOKLA_HTTP_URL = "https://ookla-open-data.s3.us-west-2.amazonaws.com/parquet/performance/type=mobile/year=2025/quarter=3/2025-07-01_performance_mobile_tiles.parquet"
LOCAL_PARQUET_PATH = "2025-07-01_performance_mobile_tiles.parquet"

# MetLife Stadium polygon (approximate boundary from aerial imagery)
# Stadium oriented roughly NW-SE
METLIFE_STADIUM_POLYGON = [
    [-74.0780, 40.8118],  # SW corner
    [-74.0780, 40.8158],  # NW corner
    [-74.0705, 40.8158],  # NE corner
    [-74.0705, 40.8118],  # SE corner
    [-74.0780, 40.8118],  # Close polygon
]


def download_ookla_data():
    """Download Ookla parquet file from S3."""
    if os.path.exists(LOCAL_PARQUET_PATH):
        file_size = os.path.getsize(LOCAL_PARQUET_PATH) / (1024 * 1024)
        print(f"Parquet file already exists ({file_size:.1f} MB)")
        return LOCAL_PARQUET_PATH

    print("Downloading Ookla Q3 2025 mobile performance data...")
    print(f"URL: {OOKLA_HTTP_URL}")

    # Try AWS CLI first
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', OOKLA_S3_PATH, LOCAL_PARQUET_PATH, '--no-sign-request'],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            file_size = os.path.getsize(LOCAL_PARQUET_PATH) / (1024 * 1024)
            print(f"Downloaded successfully via AWS CLI ({file_size:.1f} MB)")
            return LOCAL_PARQUET_PATH
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to urllib
    import urllib.request
    print("Downloading via HTTP (this may take a while for ~185MB file)...")

    try:
        urllib.request.urlretrieve(OOKLA_HTTP_URL, LOCAL_PARQUET_PATH)
        file_size = os.path.getsize(LOCAL_PARQUET_PATH) / (1024 * 1024)
        print(f"Downloaded successfully ({file_size:.1f} MB)")
        return LOCAL_PARQUET_PATH
    except Exception as e:
        print(f"Download failed: {e}")
        raise RuntimeError("Failed to download Ookla data. Please download manually.")


def quadkey_to_tile(quadkey):
    """Convert quadkey to tile coordinates (x, y, zoom)."""
    x = y = 0
    zoom = len(quadkey)
    for i, char in enumerate(quadkey):
        bit = zoom - i - 1
        mask = 1 << bit
        if char == '1':
            x |= mask
        elif char == '2':
            y |= mask
        elif char == '3':
            x |= mask
            y |= mask
    return x, y, zoom


def tile_to_bbox(x, y, zoom):
    """Convert tile coordinates to bounding box in lat/lon."""
    import math
    n = 2 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


def quadkey_to_bbox(quadkey):
    """Convert quadkey to bounding box."""
    x, y, zoom = quadkey_to_tile(quadkey)
    return tile_to_bbox(x, y, zoom)


def quadkey_to_center(quadkey):
    """Convert quadkey to center point (lat, lon)."""
    lon_min, lat_min, lon_max, lat_max = quadkey_to_bbox(quadkey)
    return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2


def get_quadkeys_for_area(center_lat, center_lon, radius_km, zoom=16):
    """Get all quadkeys that might intersect with a circular area."""
    import math

    # Calculate approximate lat/lon bounds for the search area
    lat_offset = radius_km / 111.0  # 1 degree lat ~ 111 km
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset

    return min_lat, max_lat, min_lon, max_lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    import math
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def load_and_filter_ookla_data(parquet_path):
    """Load Ookla parquet and filter for MetLife Stadium area using DuckDB."""
    print(f"\nLoading and filtering Ookla data with DuckDB...")

    # Calculate search bounds
    min_lat, max_lat, min_lon, max_lon = get_quadkeys_for_area(
        METLIFE_LAT, METLIFE_LON, SEARCH_RADIUS_KM
    )

    print(f"Search bounds:")
    print(f"  Latitude:  {min_lat:.6f} to {max_lat:.6f}")
    print(f"  Longitude: {min_lon:.6f} to {max_lon:.6f}")

    con = duckdb.connect()

    # First, let's check the schema
    schema_query = f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    try:
        schema = con.execute(schema_query).fetchdf()
        print(f"\nParquet schema:")
        print(schema)
    except Exception as e:
        print(f"Could not read schema: {e}")

    # Ookla data typically uses quadkeys at zoom level 16
    # We need to find quadkeys that fall within our search area
    # The quadkey prefix for this area at zoom 16 starts with certain digits

    # Query to get all data and filter by quadkey bounds
    # Ookla tiles are at zoom level 16, so quadkeys are 16 characters long
    query = f"""
    SELECT
        quadkey,
        avg_d_kbps,
        avg_u_kbps,
        avg_lat_ms,
        tests,
        devices
    FROM read_parquet('{parquet_path}')
    WHERE quadkey IS NOT NULL
    """

    print("\nQuerying parquet file (this may take a moment)...")

    try:
        # First get a sample to understand the data
        sample_query = f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        LIMIT 10
        """
        sample_df = con.execute(sample_query).fetchdf()
        print(f"\nSample data columns: {list(sample_df.columns)}")
        print(f"\nSample records:")
        print(sample_df.head())

        # Get total count
        count_query = f"SELECT COUNT(*) as cnt FROM read_parquet('{parquet_path}')"
        total_count = con.execute(count_query).fetchone()[0]
        print(f"\nTotal records in parquet: {total_count:,}")

        # Get all data with quadkeys
        all_data = con.execute(query).fetchdf()
        print(f"Records with quadkeys: {len(all_data):,}")

    except Exception as e:
        print(f"Error querying data: {e}")
        con.close()
        return None

    con.close()

    # Now filter for MetLife Stadium area
    print("\nFiltering for MetLife Stadium area...")

    filtered_records = []
    for _, row in all_data.iterrows():
        try:
            quadkey = str(row['quadkey'])
            center_lat, center_lon = quadkey_to_center(quadkey)

            # Check if within search radius
            distance = haversine_distance(METLIFE_LAT, METLIFE_LON, center_lat, center_lon)
            if distance <= SEARCH_RADIUS_KM:
                filtered_records.append({
                    'quadkey': quadkey,
                    'lat': center_lat,
                    'lon': center_lon,
                    'avg_d_kbps': row['avg_d_kbps'],
                    'avg_u_kbps': row['avg_u_kbps'],
                    'avg_lat_ms': row['avg_lat_ms'],
                    'tests': row['tests'],
                    'devices': row['devices'],
                    'distance_km': distance,
                    'download_mbps': row['avg_d_kbps'] / 1000 if pd.notna(row['avg_d_kbps']) else None,
                    'upload_mbps': row['avg_u_kbps'] / 1000 if pd.notna(row['avg_u_kbps']) else None
                })
        except Exception as e:
            continue

    df = pd.DataFrame(filtered_records)
    print(f"Found {len(df)} tiles within {SEARCH_RADIUS_KM} km of MetLife Stadium")

    return df


def create_mapbox_visualization(df):
    """Create interactive Mapbox visualization with stadium overlay."""
    print("\nCreating Mapbox visualization...")

    if df is None or len(df) == 0:
        print("No data to visualize!")
        return None

    # Create the main figure with Ookla data points
    fig = go.Figure()

    # Add stadium polygon outline
    stadium_lons = [p[0] for p in METLIFE_STADIUM_POLYGON]
    stadium_lats = [p[1] for p in METLIFE_STADIUM_POLYGON]

    fig.add_trace(go.Scattermapbox(
        name="MetLife Stadium",
        mode="lines",
        lon=stadium_lons,
        lat=stadium_lats,
        line=dict(width=3, color='blue'),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        hoverinfo='name'
    ))

    # Add stadium center marker
    fig.add_trace(go.Scattermapbox(
        name="Stadium Center",
        mode="markers+text",
        lon=[METLIFE_LON],
        lat=[METLIFE_LAT],
        marker=dict(size=15, color='red', symbol='stadium'),
        text=["MetLife Stadium"],
        textposition="top center",
        hovertemplate="<b>MetLife Stadium</b><br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>"
    ))

    # Color scale for download speed
    if len(df) > 0:
        # Add Ookla data points colored by download speed
        fig.add_trace(go.Scattermapbox(
            name="Ookla Speed Tests",
            mode="markers",
            lon=df['lon'],
            lat=df['lat'],
            marker=dict(
                size=12,
                color=df['download_mbps'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=df['download_mbps'].quantile(0.95) if len(df) > 0 else 100,
                colorbar=dict(
                    title="Download<br>Speed (Mbps)",
                    x=1.02
                ),
                opacity=0.8
            ),
            text=df.apply(lambda r: f"Download: {r['download_mbps']:.1f} Mbps<br>"
                                    f"Upload: {r['upload_mbps']:.1f} Mbps<br>"
                                    f"Latency: {r['avg_lat_ms']:.0f} ms<br>"
                                    f"Tests: {r['tests']}<br>"
                                    f"Devices: {r['devices']}<br>"
                                    f"Distance: {r['distance_km']*1000:.0f}m", axis=1),
            hovertemplate="<b>Ookla Tile</b><br>%{text}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>"
        ))

    # Configure the map layout
    fig.update_layout(
        title=dict(
            text="<b>MetLife Stadium - Ookla Q3 2025 Mobile Performance Data</b>",
            x=0.5,
            font=dict(size=20)
        ),
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style="mapbox://styles/mapbox/satellite-streets-v12",
            center=dict(lat=METLIFE_LAT, lon=METLIFE_LON),
            zoom=15,
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=800
    )

    # Add annotations
    if len(df) > 0:
        avg_download = df['download_mbps'].mean()
        avg_upload = df['upload_mbps'].mean()
        avg_latency = df['avg_lat_ms'].mean()
        total_tests = df['tests'].sum()

        stats_text = (
            f"<b>Statistics (Q3 2025):</b><br>"
            f"Tiles: {len(df)}<br>"
            f"Avg Download: {avg_download:.1f} Mbps<br>"
            f"Avg Upload: {avg_upload:.1f} Mbps<br>"
            f"Avg Latency: {avg_latency:.0f} ms<br>"
            f"Total Tests: {total_tests:,}"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.15,
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1
        )

    return fig


def create_detailed_analysis_html(df, fig):
    """Create comprehensive HTML report with analysis and visualization."""
    if df is None:
        df = pd.DataFrame()

    # Statistics
    if len(df) > 0:
        stats = {
            'total_tiles': len(df),
            'avg_download': df['download_mbps'].mean(),
            'max_download': df['download_mbps'].max(),
            'min_download': df['download_mbps'].min(),
            'avg_upload': df['upload_mbps'].mean(),
            'max_upload': df['upload_mbps'].max(),
            'avg_latency': df['avg_lat_ms'].mean(),
            'min_latency': df['avg_lat_ms'].min(),
            'total_tests': df['tests'].sum(),
            'total_devices': df['devices'].sum()
        }
    else:
        stats = {}

    # Convert figure to HTML div
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn') if fig else ""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MetLife Stadium - Ookla Q3 2025 Mobile Performance Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        .header {{
            background: linear-gradient(90deg, #0066cc 0%, #0099ff 100%);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 153, 255, 0.3);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #ccc;
            margin-top: 5px;
        }}
        .map-container {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin: 30px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .map-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #00ff88;
        }}
        .info-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
        }}
        .info-section h2 {{
            color: #00ff88;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        .info-section p {{
            line-height: 1.8;
            color: #ddd;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .data-table th {{
            background: rgba(0, 153, 255, 0.3);
            color: #fff;
        }}
        .data-table tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #888;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 50px;
        }}
        .download-badge {{
            background: linear-gradient(45deg, #00ff88, #00cc66);
            color: #000;
            padding: 2px 8px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .upload-badge {{
            background: linear-gradient(45deg, #0099ff, #0066cc);
            color: #fff;
            padding: 2px 8px;
            border-radius: 20px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MetLife Stadium Network Analysis</h1>
        <p>Ookla Speedtest Q3 2025 Mobile Performance Data</p>
        <p style="margin-top: 10px; font-size: 0.9em;">Coordinates: 40.813778&deg;N, 74.074310&deg;W | East Rutherford, NJ</p>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_tiles', 'N/A')}</div>
                <div class="stat-label">Coverage Tiles</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('avg_download', 0):.1f}</div>
                <div class="stat-label">Avg Download (Mbps)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('max_download', 0):.1f}</div>
                <div class="stat-label">Max Download (Mbps)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('avg_upload', 0):.1f}</div>
                <div class="stat-label">Avg Upload (Mbps)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('avg_latency', 0):.0f}</div>
                <div class="stat-label">Avg Latency (ms)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_tests', 0):,}</div>
                <div class="stat-label">Total Speed Tests</div>
            </div>
        </div>

        <div class="map-container">
            <div class="map-title">Interactive Performance Map</div>
            {plot_div}
        </div>

        <div class="info-section">
            <h2>About This Analysis</h2>
            <p>
                This analysis visualizes mobile network performance data from Ookla Speedtest
                for Q3 2025 (July-September) in and around MetLife Stadium. The data represents
                aggregated speed test results from Speedtest mobile applications.
            </p>
            <p style="margin-top: 15px;">
                <strong>Data Source:</strong> Ookla Open Data Initiative<br>
                <strong>Coverage Area:</strong> 500m radius from stadium center<br>
                <strong>Tile Resolution:</strong> Zoom level 16 (~610m x 610m tiles at this latitude)<br>
                <strong>Time Period:</strong> Q3 2025 (July 1 - September 30, 2025)
            </p>
        </div>

        {"" if len(df) == 0 else f'''
        <div class="info-section">
            <h2>Detailed Tile Data</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Quadkey</th>
                        <th>Download</th>
                        <th>Upload</th>
                        <th>Latency</th>
                        <th>Tests</th>
                        <th>Distance</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([f'''
                    <tr>
                        <td><code>{row['quadkey'][:8]}...</code></td>
                        <td><span class="download-badge">{row['download_mbps']:.1f} Mbps</span></td>
                        <td><span class="upload-badge">{row['upload_mbps']:.1f} Mbps</span></td>
                        <td>{row['avg_lat_ms']:.0f} ms</td>
                        <td>{row['tests']:,}</td>
                        <td>{row['distance_km']*1000:.0f}m</td>
                    </tr>
                    ''' for _, row in df.head(20).iterrows()])}
                </tbody>
            </table>
            <p style="margin-top: 15px; color: #888; font-size: 0.9em;">
                Showing top 20 tiles. Total tiles: {len(df)}
            </p>
        </div>
        '''}

        <div class="footer">
            <p>Data: Ookla Open Data | Visualization: Plotly + Mapbox</p>
            <p style="margin-top: 10px;">Generated for MetLife Stadium Network Performance Analysis</p>
            <p style="margin-top: 5px; font-size: 0.8em;">
                Mapbox Token: {MAPBOX_TOKEN[:20]}...
            </p>
        </div>
    </div>
</body>
</html>
"""
    return html_content


def main():
    """Main execution function."""
    print("=" * 60)
    print("MetLife Stadium - Ookla Q3 2025 Mobile Performance Analysis")
    print("=" * 60)
    print(f"\nStadium Coordinates: {METLIFE_LAT}, {METLIFE_LON}")
    print(f"Search Radius: {SEARCH_RADIUS_KM * 1000}m")

    # Step 1: Download Ookla data
    try:
        parquet_path = download_ookla_data()
    except Exception as e:
        print(f"\nError downloading data: {e}")
        print("\nCreating visualization with placeholder for data...")
        parquet_path = None

    # Step 2: Load and filter data
    df = None
    if parquet_path and os.path.exists(parquet_path):
        df = load_and_filter_ookla_data(parquet_path)

    # Step 3: Create visualization
    fig = create_mapbox_visualization(df)

    # Step 4: Generate HTML file
    html_content = create_detailed_analysis_html(df, fig)

    output_file = "metlife_stadium_ookla_q3_2025.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
    print(f"HTML file saved: {output_file}")
    print(f"{'=' * 60}")

    # Also save the raw figure as standalone HTML
    if fig:
        fig.write_html("metlife_stadium_map.html")
        print(f"Map-only file saved: metlife_stadium_map.html")

    return df, fig


if __name__ == "__main__":
    df, fig = main()
