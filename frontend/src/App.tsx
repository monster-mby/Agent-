import { Sidebar } from '@/components/Sidebar'
import { ChatArea } from '@/components/ChatArea'
import { ContextPanel } from '@/components/ContextPanel'
import { useUIStore } from '@/stores/uiStore'
import { useMediaQuery } from '@/hooks/useResponsive'
import { Menu, X } from 'lucide-react'

function App() {
  const isMobile = useMediaQuery('(max-width: 768px)')
  const isTablet = useMediaQuery('(max-width: 1024px)')
  const { sidebarOpen, toggleSidebar } = useUIStore()

  return (
    <div className="h-screen flex relative">
      {isMobile && (
        <button
          onClick={toggleSidebar}
          className="fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow"
        >
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      )}

      <aside
        className={`
          ${isMobile ? 'fixed inset-y-0 left-0 z-40 w-72 transform transition-transform' : 'w-72'}
          ${isMobile && !sidebarOpen ? '-translate-x-full' : ''}
        `}
      >
        <Sidebar />
      </aside>

      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={toggleSidebar}
        />
      )}

      <main className={`flex-1 ${isMobile ? 'pt-12' : ''}`}>
        <ChatArea />
      </main>

      {!isTablet && (
        <aside className="w-80">
          <ContextPanel />
        </aside>
      )}
    </div>
  )
}

export default App
