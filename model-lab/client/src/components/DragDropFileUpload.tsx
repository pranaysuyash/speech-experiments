import { useState, useRef, useCallback } from 'react';
import { Upload, X, FileAudio, FileVideo } from 'lucide-react';

interface DragDropFileUploadProps {
  accept: string;
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
  maxSizeBytes?: number;
}

export default function DragDropFileUpload({
  accept,
  onFileSelect,
  selectedFile,
  disabled = false,
  maxSizeBytes,
}: DragDropFileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback(
    (file: File): boolean => {
      setError(null);

      // Check file size
      if (maxSizeBytes && file.size > maxSizeBytes) {
        setError(
          `File size exceeds ${Math.round(maxSizeBytes / (1024 * 1024))}MB limit`,
        );
        return false;
      }

      // Check file type
      const acceptedTypes = accept.split(',').map((type) => type.trim());
      const fileType = file.type.toLowerCase();
      const fileName = file.name.toLowerCase();

      const isAccepted = acceptedTypes.some((acceptedType) => {
        if (acceptedType.startsWith('.')) {
          // Extension check
          return fileName.endsWith(acceptedType.toLowerCase());
        } else if (acceptedType.includes('*')) {
          // MIME type pattern
          const pattern = acceptedType.replace('*', '.*');
          return new RegExp(pattern).test(fileType);
        } else {
          // Exact MIME type
          return fileType === acceptedType.toLowerCase();
        }
      });

      if (!isAccepted) {
        setError(
          'File type not supported. Please select an audio or video file.',
        );
        return false;
      }

      return true;
    },
    [accept, maxSizeBytes],
  );

  const handleFileSelect = useCallback(
    (file: File | null) => {
      if (file && validateFile(file)) {
        onFileSelect(file);
      } else if (!file) {
        onFileSelect(null);
        setError(null);
      }
    },
    [onFileSelect, validateFile],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) {
        setIsDragOver(true);
      }
    },
    [disabled],
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    },
    [disabled, handleFileSelect],
  );

  const handleClick = useCallback(() => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [disabled]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] || null;
      handleFileSelect(file);
    },
    [handleFileSelect],
  );

  const handleRemove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onFileSelect(null);
      setError(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [onFileSelect],
  );

  const getFileIcon = (file: File) => {
    const type = file.type.toLowerCase();
    if (type.startsWith('audio/')) {
      return <FileAudio className='w-8 h-8 text-blue-500' />;
    } else if (type.startsWith('video/')) {
      return <FileVideo className='w-8 h-8 text-purple-500' />;
    }
    return <Upload className='w-8 h-8 text-gray-400' />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className='w-full'>
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-6 transition-all cursor-pointer
          ${
            disabled
              ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
              : isDragOver
                ? 'border-blue-400 bg-blue-50'
                : selectedFile
                  ? 'border-green-300 bg-green-50'
                  : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
          }
        `}
      >
        <input
          ref={fileInputRef}
          type='file'
          accept={accept}
          onChange={handleInputChange}
          disabled={disabled}
          className='hidden'
        />

        <div className='text-center'>
          {selectedFile ? (
            <div className='flex items-center justify-center space-x-3'>
              {getFileIcon(selectedFile)}
              <div className='flex-1 text-left'>
                <p className='font-medium text-gray-900 truncate'>
                  {selectedFile.name}
                </p>
                <p className='text-sm text-gray-500'>
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
              {!disabled && (
                <button
                  onClick={handleRemove}
                  className='p-1 hover:bg-gray-200 rounded-full transition-colors'
                  title='Remove file'
                >
                  <X className='w-4 h-4 text-gray-500' />
                </button>
              )}
            </div>
          ) : (
            <div>
              <Upload
                className={`w-12 h-12 mx-auto mb-4 ${disabled ? 'text-gray-300' : 'text-gray-400'}`}
              />
              <p
                className={`text-lg font-medium mb-1 ${disabled ? 'text-gray-400' : 'text-gray-900'}`}
              >
                {isDragOver
                  ? 'Drop your file here'
                  : 'Drag & drop your file here'}
              </p>
              <p
                className={`text-sm ${disabled ? 'text-gray-400' : 'text-gray-500'}`}
              >
                or{' '}
                <span className='text-blue-500 hover:text-blue-600 font-medium'>
                  browse files
                </span>
              </p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className='mt-2 text-red-600 text-sm font-medium'>{error}</div>
      )}

      <div className='mt-2 text-xs text-gray-500'>
        Accepted: Audio/Video files (WAV, MP3, M4A, MP4, MOV, AVI, etc.)
        {maxSizeBytes &&
          ` • Max size: ${Math.round(maxSizeBytes / (1024 * 1024))}MB`}
      </div>
    </div>
  );
}
