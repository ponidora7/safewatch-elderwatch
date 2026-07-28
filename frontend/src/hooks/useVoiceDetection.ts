import { useState, useEffect, useCallback, useRef } from 'react';

export interface VoiceDetectionResult {
  detected: boolean;
  keyword: string | null;
  confidence: number;
}

interface UseVoiceDetectionOptions {
  enabled?: boolean;
  keywords?: string[];
  lang?: string;
  onDetect?: (result: VoiceDetectionResult) => void;
}

export function useVoiceDetection({
  enabled = true,
  keywords = ['tolong', 'bantu', 'sakit', 'jatuh', 'aduh'],
  lang = 'id-ID',
  onDetect
}: UseVoiceDetectionOptions = {}) {
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastDetected, setLastDetected] = useState<string | null>(null);
  
  const recognitionRef = useRef<any>(null);
  const isEnabledRef = useRef(enabled);

  // Keep ref in sync
  useEffect(() => {
    isEnabledRef.current = enabled;
  }, [enabled]);

  useEffect(() => {
    // Check browser support
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError('Browser tidak mendukung fitur Speech Recognition.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = lang;
    
    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onresult = (event: any) => {
      if (!isEnabledRef.current) return;
      
      const current = event.resultIndex;
      const transcript = event.results[current][0].transcript.toLowerCase();
      
      // Check for keywords
      const foundKeyword = keywords.find(k => transcript.includes(k));
      
      if (foundKeyword && event.results[current].isFinal) {
        setLastDetected(foundKeyword);
        const result: VoiceDetectionResult = {
          detected: true,
          keyword: foundKeyword,
          confidence: event.results[current][0].confidence || 0.8
        };
        
        if (onDetect) {
          onDetect(result);
        }
        
        // Reset last detected after 5 seconds to clear UI
        setTimeout(() => {
          setLastDetected(null);
        }, 5000);
      }
    };

    recognition.onerror = (event: any) => {
      if (event.error === 'not-allowed') {
        setError('Akses mikrofon ditolak.');
        setIsListening(false);
      } else if (event.error !== 'no-speech') {
        // no-speech is common when quiet, ignore it
        console.warn('Speech recognition error', event.error);
      }
    };

    recognition.onend = () => {
      // Auto-restart if it stopped but is still enabled
      if (isEnabledRef.current) {
        try {
          recognition.start();
        } catch (e) {
          setIsListening(false);
        }
      } else {
        setIsListening(false);
      }
    };

    recognitionRef.current = recognition;

    if (enabled) {
      try {
        recognition.start();
      } catch (e) {
        console.error("Failed to start speech recognition", e);
      }
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null; // Prevent restart on unmount
        recognitionRef.current.stop();
      }
    };
  }, [lang, keywords, onDetect]);

  // Handle runtime toggle
  useEffect(() => {
    if (!recognitionRef.current) return;
    
    if (enabled && !isListening) {
      try {
        recognitionRef.current.start();
      } catch (e) {}
    } else if (!enabled && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [enabled, isListening]);

  return {
    isListening,
    error,
    lastDetected
  };
}
