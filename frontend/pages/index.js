import { useState } from 'react'

export default function Home() {
  const [message, setMessage] = useState('')
  const [reply, setReply] = useState('')
  const [loading, setLoading] = useState(false)

  async function send() {
    setLoading(true)
    setReply('')
    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
      const data = await res.json()
      if (res.ok) {
        setReply(data.reply)
      } else {
        setReply('Error: ' + (data.detail || JSON.stringify(data)))
      }
    } catch (e) {
      setReply('Request failed: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h1>RezumAI — Frontend (Next.js)</h1>
      <p>Type a message and hit send. Backend will respond via Vertex AI (placeholder).</p>

      <textarea
        rows={6}
        style={{ width: '100%', fontSize: 16 }}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Enter a message for Vertex AI..."
      />

      <div style={{ marginTop: 12 }}>
        <button onClick={send} disabled={loading || !message} style={{ padding: '8px 16px' }}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div style={{ marginTop: 18 }}>
        <h3>Reply</h3>
        <pre style={{ whiteSpace: 'pre-wrap', background: '#f6f6f6', padding: 12 }}>{reply}</pre>
      </div>
    </div>
  )
}
