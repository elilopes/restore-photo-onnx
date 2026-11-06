import React, { useState } from 'react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    setStatus('Enviando para o pipeline...');
    try {
      const res = await fetch('http://localhost:3000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setStatus(`Sucesso! Tarefa enfileirada com ID: ${data.jobId}`);
    } catch (err) {
      setStatus('Erro ao conectar com a API.');
    }
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'Segoe UI, sans-serif', maxWidth: '600px', margin: 'auto' }}>
      <h2>Restauração de Fotos Antigas (ONNX + OpenCV)</h2>
      <form onSubmit={handleUpload} style={{ marginTop: '20px' }}>
        <input 
          type="file" 
          onChange={(e) => setSelectedFile(e.target.files[0])} 
          style={{ display: 'block', marginBottom: '15px' }}
        />
        <button 
          type="submit" 
          style={{ padding: '8px 16px', background: '#007acc', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Enviar Foto
        </button>
      </form>
      <p style={{ marginTop: '15px', fontWeight: 'bold', color: '#333' }}>{status}</p>
    </div>
  );
}