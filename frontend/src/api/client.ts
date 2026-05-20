import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器：注入 API Key
apiClient.interceptors.request.use((config) => {
  const apiKey = import.meta.env.VITE_API_KEY || ''
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})
