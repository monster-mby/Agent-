import { Settings } from 'lucide-react'

export const ContextPanel = () => {
  return (
    <div className="h-full bg-white border-l flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-semibold flex items-center gap-2">
          <Settings size={18} />
          上下文设置
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">知识库</h3>
            <p className="text-xs text-gray-500">暂无知识库</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">规则</h3>
            <p className="text-xs text-gray-500">暂无规则</p>
          </div>
        </div>
      </div>
    </div>
  )
}
