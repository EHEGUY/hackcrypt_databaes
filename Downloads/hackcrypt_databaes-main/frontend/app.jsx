"use client"

import { useState, useRef, useEffect } from "react"

function App() {
  const [media, setMedia] = useState(null)
  const [isScanning, setIsScanning] = useState(false)
  const [result, setResult] = useState(null)
  const [view, setView] = useState("home")
  const fileInputRef = useRef(null)

  // This forces the body to be black and centered, overriding Vite's defaults
  useEffect(() => {
    document.body.style.margin = "0"
    document.body.style.padding = "0"
    document.body.style.backgroundColor = "#000"
    document.body.style.display = "block"
  }, [])

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      if (media?.url) {
        URL.revokeObjectURL(media.url)
      }
      const url = URL.createObjectURL(file)
      setMedia({
        url: url,
        type: file.type.startsWith("video") ? "video" : "image",
        name: file.name,
      })
      setResult(null)
    }
  }

  const startScan = () => {
    setIsScanning(true)
    setResult(null)
    setTimeout(() => {
      const isFake = Math.random() > 0.5
      setResult({
        status: isFake ? "Fake" : "Real",
        confidence: (85 + Math.random() * 14).toFixed(1),
        isFake: isFake,
      })
      setIsScanning(false)
    }, 2000)
  }

  const reset = () => {
    if (media?.url) {
      URL.revokeObjectURL(media.url)
    }
    setMedia(null)
    setResult(null)
    setIsScanning(false)
  }

  return (
    <div
      style={{
        backgroundColor: "#000",
        color: "#fff",
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif',
        padding: "20px",
        margin: "0",
        position: "relative",
        overflowX: "hidden",
      }}
    >
      {/* Navigation - Fixed to top */}
      <nav
        style={{
          position: "absolute",
          top: "40px",
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: "30px",
          zIndex: 100,
        }}
      >
        <span
          onClick={() => setView("home")}
          style={{
            cursor: "pointer",
            color: view === "home" ? "#fff" : "#666",
            fontSize: "14px",
            fontWeight: "500",
            transition: "color 0.3s ease",
          }}
        >
          Verify
        </span>
        <span
          onClick={() => setView("about")}
          style={{
            cursor: "pointer",
            color: view === "about" ? "#fff" : "#666",
            fontSize: "14px",
            fontWeight: "500",
            transition: "color 0.3s ease",
          }}
        >
          About
        </span>
      </nav>

      <div
        style={{
          width: "100%",
          maxWidth: "500px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          marginTop: "60px",
        }}
      >
        {view === "home" ? (
          <>
            <h1
              style={{
                fontSize: "3.5rem",
                fontWeight: "600",
                textAlign: "center",
                marginBottom: "40px",
                letterSpacing: "-2px",
              }}
            >
              Verify
            </h1>

            {!media ? (
              <div
                onClick={() => fileInputRef.current.click()}
                style={{
                  backgroundColor: "#111",
                  borderRadius: "28px",
                  padding: "80px 20px",
                  textAlign: "center",
                  cursor: "pointer",
                  border: "1px solid #222",
                  width: "100%",
                  transition: "all 0.3s ease",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#1a1a1a")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#111")}
              >
                <div style={{ fontSize: "40px", marginBottom: "10px" }}>📁</div>
                <h3 style={{ fontSize: "1.2rem", color: "#fff", marginBottom: "5px", fontWeight: "500" }}>
                  Upload media
                </h3>
                <p style={{ color: "#666", fontSize: "0.8rem" }}>MP4, MOV, JPG, PNG</p>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/*,image/*" hidden />
              </div>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "24px",
                  width: "100%",
                }}
              >
                <div
                  style={{
                    backgroundColor: "#0a0a0a",
                    borderRadius: "24px",
                    overflow: "hidden",
                    border: "1px solid #222",
                    width: "100%",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    minHeight: "300px",
                  }}
                >
                  {media.type === "video" ? (
                    <video
                      key={media.url}
                      controls
                      playsInline
                      preload="metadata"
                      style={{
                        width: "100%",
                        height: "auto",
                        maxHeight: "60vh",
                        display: "block",
                      }}
                    >
                      <source src={media.url} type="video/mp4" />
                      <source src={media.url} type="video/quicktime" />
                      Your browser does not support the video tag.
                    </video>
                  ) : (
                    <img
                      src={media.url || "/placeholder.svg"}
                      alt="Preview"
                      style={{ width: "100%", maxHeight: "450px", objectFit: "contain" }}
                    />
                  )}
                </div>

                {/* Scan Button */}
                <button
                  onClick={startScan}
                  style={{
                    backgroundColor: "#111",
                    color: "#fff",
                    padding: "10px 20px",
                    borderRadius: "28px",
                    border: "none",
                    cursor: "pointer",
                    transition: "background-color 0.3s ease",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#1a1a1a")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#111")}
                >
                  Scan
                </button>

                {/* Result Display */}
                {result && (
                  <div
                    style={{
                      marginTop: "20px",
                      padding: "10px",
                      backgroundColor: result.isFake ? "#ff6347" : "#7cfc00",
                      borderRadius: "10px",
                      color: "#fff",
                      textAlign: "center",
                    }}
                  >
                    <h2 style={{ margin: "0", fontSize: "1.5rem" }}>{result.status}</h2>
                    <p style={{ margin: "5px 0", fontSize: "0.9rem" }}>Confidence: {result.confidence}%</p>
                  </div>
                )}

                {/* Reset Button */}
                <button
                  onClick={reset}
                  style={{
                    marginTop: "20px",
                    backgroundColor: "#111",
                    color: "#fff",
                    padding: "10px 20px",
                    borderRadius: "28px",
                    border: "none",
                    cursor: "pointer",
                    transition: "background-color 0.3s ease",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#1a1a1a")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#111")}
                >
                  Reset
                </button>
              </div>
            )}
          </>
        ) : (
          <div
            style={{
              textAlign: "center",
              marginTop: "40px",
            }}
          >
            <h2 style={{ fontSize: "2rem", marginBottom: "20px" }}>About</h2>
            <p style={{ fontSize: "1rem", color: "#666" }}>
              This application allows you to upload media files and verify their authenticity.
            </p>
          </div>
        )}
      </div>

      <style>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        .scan-bar {
          animation: loading 2s infinite linear;
        }
        * {
          box-sizing: border-box;
        }
        body, html {
          margin: 0;
          padding: 0;
          background-color: #000;
          overflow-x: hidden;
        }
      `}</style>
    </div>
  )
}

export default App
