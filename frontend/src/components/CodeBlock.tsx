import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodeBlockProps {
  language: string
  code: string
}

export const CodeBlock = ({ language, code }: CodeBlockProps) => {
  return (
    <div className="relative rounded-lg overflow-hidden my-2">
      {/* 语言标签 */}
      <div className="bg-gray-700 text-gray-300 px-4 py-1 text-xs font-mono">
        {language}
      </div>

      {/* 代码高亮 */}
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={{ margin: 0, borderRadius: 0 }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}
