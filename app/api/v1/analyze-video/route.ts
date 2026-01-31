export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    
    // Forward the request to the Python backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/api/v1/analyze-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Analysis failed' }));
      return Response.json(errorData, { status: response.status });
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('[v0] Backend error:', error);
    return Response.json(
      { error: 'Failed to analyze video. Please ensure the backend is running.' },
      { status: 500 }
    );
  }
}
