import { useState, useEffect, useRef } from 'react'
import { Box, TextField, IconButton, Typography, Paper, CircularProgress, Divider } from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import MicIcon from '@mui/icons-material/Mic'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'

const Chat = () => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const messagesEndRef = useRef(null)
  
  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [messages])
  
  // Add welcome message on first load
  useEffect(() => {
    setMessages([
      { 
        role: 'assistant', 
        content: 'Hello, I\'m Elara, your medical AI assistant. How can I help you today? You can ask me medical questions, search for drug information, or help find doctors.'
      }
    ])
  }, [])
  
  // Send text message to API
  const handleSendMessage = async (e) => {
    e.preventDefault()
    
    if (!input.trim()) return
    
    // Add user message to chat
    const userMessage = { role: 'user', content: input }
    setMessages(prevMessages => [...prevMessages, userMessage])
    setInput('')
    setIsLoading(true)
    
    try {
      // Send request to backend API
      const response = await axios.post('/api/ask', {
        question: input,
        language: 'en', // This could be dynamic based on user selection
      })
      
      // Add assistant response to chat
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          role: 'assistant', 
          content: response.data.answer,
          sources: response.data.sources
        }
      ])
      
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          role: 'assistant', 
          content: 'I\'m sorry, I encountered an error processing your request. Please try again later.'
        }
      ])
    } finally {
      setIsLoading(false)
    }
  }
  
  // Handle voice input
  const handleVoiceInput = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      return
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      
      const audioChunks = []
      
      mediaRecorder.addEventListener('dataavailable', (event) => {
        audioChunks.push(event.data)
      })
      
      mediaRecorder.addEventListener('stop', async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
        
        // Convert blob to base64
        const reader = new FileReader()
        reader.readAsDataURL(audioBlob)
        reader.onloadend = async () => {
          const base64Audio = reader.result.split(',')[1]
          
          setIsLoading(true)
          
          try {
            // Send audio to transcription API
            const response = await axios.post('/api/voice', {
              audio_data: base64Audio,
              language: 'en' // This could be dynamic
            })
            
            const transcription = response.data.transcription
            
            // Add transcription as user message
            if (transcription) {
              setInput(transcription)
              // Automatically send the transcribed message
              const userMessage = { role: 'user', content: transcription }
              setMessages(prevMessages => [...prevMessages, userMessage])
              
              // Get AI response
              const aiResponse = await axios.post('/api/ask', {
                question: transcription,
                language: 'en'
              })
              
              // Add assistant response
              setMessages(prevMessages => [
                ...prevMessages, 
                { 
                  role: 'assistant', 
                  content: aiResponse.data.answer,
                  sources: aiResponse.data.sources
                }
              ])
            }
          } catch (error) {
            console.error('Error processing voice input:', error)
            setMessages(prevMessages => [
              ...prevMessages, 
              { 
                role: 'assistant', 
                content: 'I\'m sorry, I encountered an error processing your voice input. Please try again.'
              }
            ])
          } finally {
            setIsLoading(false)
            setInput('')
          }
        }
        
        // Stop all audio tracks
        stream.getTracks().forEach(track => track.stop())
      })
      
      // Start recording
      mediaRecorder.start()
      setIsRecording(true)
      
    } catch (error) {
      console.error('Error accessing microphone:', error)
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          role: 'assistant', 
          content: 'I\'m sorry, I couldn\'t access your microphone. Please check your browser permissions.'
        }
      ])
    }
  }
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 2 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Elara Medical Assistant
      </Typography>
      
      {/* Messages container */}
      <Box sx={{ 
        flex: 1, 
        overflowY: 'auto', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: 2,
        mb: 2,
        p: 2,
        bgcolor: 'background.default',
        borderRadius: 2
      }}>
        {messages.map((message, index) => (
          <Paper 
            key={index} 
            elevation={1} 
            sx={{
              p: 2,
              maxWidth: '80%',
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
              bgcolor: message.role === 'user' ? 'primary.light' : 'background.paper',
              color: message.role === 'user' ? 'primary.contrastText' : 'text.primary'
            }}
          >
            <ReactMarkdown>
              {message.content}
            </ReactMarkdown>
            
            {message.sources && message.sources.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Divider sx={{ my: 1 }} />
                <Typography variant="caption" color="text.secondary">
                  Sources:
                </Typography>
                <ul style={{ marginTop: 4, paddingLeft: 20 }}>
                  {message.sources.map((source, idx) => (
                    <li key={idx}>
                      <Typography variant="caption" color="text.secondary">
                        {source.metadata.title || source.metadata.url || `Source ${idx + 1}`}
                      </Typography>
                    </li>
                  ))}
                </ul>
              </Box>
            )}
          </Paper>
        ))}
        
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Box>
      
      {/* Input form */}
      <Box 
        component="form" 
        onSubmit={handleSendMessage}
        sx={{ 
          display: 'flex', 
          alignItems: 'center',
          gap: 1,
          p: 1,
          borderRadius: 2,
          bgcolor: 'background.paper' 
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask Elara a medical question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          sx={{ '& .MuiOutlinedInput-root': { borderRadius: 4 } }}
        />
        
        <IconButton 
          color={isRecording ? 'error' : 'primary'} 
          onClick={handleVoiceInput}
          disabled={isLoading}
        >
          <MicIcon />
        </IconButton>
        
        <IconButton 
          color="primary" 
          type="submit" 
          disabled={!input.trim() || isLoading}
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Box>
  )
}

export default Chat
