import './globals.css';

export const metadata = {
  title: 'DocuMind',
  description: 'Chat with your uploaded documents using RAG',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased">{children}</body>
    </html>
  );
}
