/**
 * Proxy server that translates Anthropic API to OpenAI/Azure format
 */
import http from 'http';
import https from 'https';
import { URL } from 'url';
import { type AzureConfig } from './config.js';

interface ProxyConfig {
  port: number;
  azure: AzureConfig;
  verbose: boolean;
}

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | Array<{ type: string; text?: string; source?: any }>;
}

interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

// Model mapping: Anthropic -> Azure deployment
function mapModel(model: string, config: AzureConfig): string {
  if (model.includes('opus')) return config.deployments.opus;
  if (model.includes('haiku')) return config.deployments.haiku;
  return config.deployments.sonnet; // Default to sonnet
}

// Convert Anthropic messages to OpenAI format
function convertMessages(messages: AnthropicMessage[], system?: string): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  if (system) {
    result.push({ role: 'system', content: system });
  }

  for (const msg of messages) {
    let content: string;
    if (typeof msg.content === 'string') {
      content = msg.content;
    } else {
      // Handle content blocks (text, images, etc.)
      content = msg.content
        .filter((block) => block.type === 'text')
        .map((block) => block.text || '')
        .join('\n');
    }
    result.push({ role: msg.role, content });
  }

  return result;
}

// Convert Anthropic tools to OpenAI function format
function convertTools(tools: any[]): any[] {
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.input_schema || { type: 'object', properties: {} },
    },
  }));
}

// Create SSE event string
function sseEvent(event: string, data: any): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function createProxy(config: ProxyConfig): http.Server {
  const { port, azure, verbose } = config;

  const server = http.createServer(async (req, res) => {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    // Health check
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
      return;
    }

    // Only handle POST to /v1/messages
    if (req.method !== 'POST' || !req.url?.includes('/messages')) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    // Read request body
    let body = '';
    for await (const chunk of req) {
      body += chunk;
    }

    try {
      const anthropicReq = JSON.parse(body);
      const isStreaming = anthropicReq.stream === true;

      if (verbose) {
        console.log(`[PROXY] ${anthropicReq.model} -> ${mapModel(anthropicReq.model, azure)}`);
      }

      // Build OpenAI request
      const openaiReq: any = {
        model: mapModel(anthropicReq.model, azure),
        messages: convertMessages(anthropicReq.messages, anthropicReq.system),
        max_tokens: anthropicReq.max_tokens || 4096,
        temperature: anthropicReq.temperature ?? 1,
        stream: isStreaming,
      };

      // Add tools if present
      if (anthropicReq.tools && anthropicReq.tools.length > 0) {
        openaiReq.tools = convertTools(anthropicReq.tools);
        openaiReq.tool_choice = 'auto';
      }

      // Make request to Azure
      const azureUrl = new URL(
        `/openai/deployments/${openaiReq.model}/chat/completions?api-version=${azure.apiVersion}`,
        azure.endpoint
      );

      const azureReqOptions: https.RequestOptions = {
        hostname: azureUrl.hostname,
        port: 443,
        path: azureUrl.pathname + azureUrl.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': azure.apiKey,
        },
      };

      if (isStreaming) {
        await handleStreaming(res, azureReqOptions, openaiReq, anthropicReq, verbose);
      } else {
        await handleNonStreaming(res, azureReqOptions, openaiReq, anthropicReq, verbose);
      }
    } catch (error: any) {
      console.error('[PROXY] Error:', error.message);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
  });

  return server;
}

async function handleStreaming(
  res: http.ServerResponse,
  options: https.RequestOptions,
  openaiReq: any,
  anthropicReq: any,
  verbose: boolean
): Promise<void> {
  return new Promise((resolve, reject) => {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });

    const msgId = `msg_${Date.now()}`;
    let inputTokens = 0;
    let outputTokens = 0;

    // Send message_start
    res.write(
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: msgId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: anthropicReq.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      })
    );

    const azureReq = https.request(options, (azureRes) => {
      let buffer = '';
      let textBlockStarted = false;
      let currentBlockIndex = 0;
      const toolCalls: Map<number, { id: string; name: string; arguments: string }> = new Map();
      const toolBlocksStarted = new Set<number>();

      azureRes.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6).trim();
          if (dataStr === '[DONE]') continue;

          try {
            const data = JSON.parse(dataStr);
            const choices = data.choices || [];

            if (!choices.length) continue;

            const choice = choices[0];
            const delta = choice.delta || {};

            // Handle finish reason
            if (choice.finish_reason) {
              // Close any open blocks
              if (textBlockStarted) {
                res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: currentBlockIndex }));
              }

              // Close tool blocks
              for (const [tcIndex] of toolCalls) {
                const blockIndex = currentBlockIndex + tcIndex + 1;
                if (toolBlocksStarted.has(blockIndex)) {
                  res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: blockIndex }));
                }
              }

              const stopReason = choice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn';
              res.write(
                sseEvent('message_delta', {
                  type: 'message_delta',
                  delta: { stop_reason: stopReason, stop_sequence: null },
                  usage: { output_tokens: outputTokens },
                })
              );
              res.write(sseEvent('message_stop', { type: 'message_stop' }));
              res.end();
              resolve();
              return;
            }

            // Handle text content
            if (delta.content) {
              if (!textBlockStarted) {
                res.write(
                  sseEvent('content_block_start', {
                    type: 'content_block_start',
                    index: currentBlockIndex,
                    content_block: { type: 'text', text: '' },
                  })
                );
                textBlockStarted = true;
              }

              res.write(
                sseEvent('content_block_delta', {
                  type: 'content_block_delta',
                  index: currentBlockIndex,
                  delta: { type: 'text_delta', text: delta.content },
                })
              );
              outputTokens++;
            }

            // Handle tool calls
            if (delta.tool_calls) {
              // Close text block if open
              if (textBlockStarted && !toolBlocksStarted.size) {
                res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: currentBlockIndex }));
                currentBlockIndex++;
                textBlockStarted = false;
              }

              for (const tc of delta.tool_calls) {
                const tcIndex = tc.index || 0;
                const blockIndex = currentBlockIndex + tcIndex;

                if (!toolCalls.has(tcIndex)) {
                  toolCalls.set(tcIndex, { id: tc.id || '', name: '', arguments: '' });
                }

                const toolCall = toolCalls.get(tcIndex)!;
                if (tc.id) toolCall.id = tc.id;
                if (tc.function?.name) toolCall.name = tc.function.name;
                if (tc.function?.arguments) toolCall.arguments += tc.function.arguments;

                // Start tool block if not started
                if (!toolBlocksStarted.has(blockIndex) && toolCall.name) {
                  res.write(
                    sseEvent('content_block_start', {
                      type: 'content_block_start',
                      index: blockIndex,
                      content_block: {
                        type: 'tool_use',
                        id: toolCall.id,
                        name: toolCall.name,
                        input: {},
                      },
                    })
                  );
                  toolBlocksStarted.add(blockIndex);
                }

                // Send argument delta
                if (tc.function?.arguments && toolBlocksStarted.has(blockIndex)) {
                  res.write(
                    sseEvent('content_block_delta', {
                      type: 'content_block_delta',
                      index: blockIndex,
                      delta: { type: 'input_json_delta', partial_json: tc.function.arguments },
                    })
                  );
                }
              }
            }

            // Handle usage
            if (data.usage) {
              inputTokens = data.usage.prompt_tokens || inputTokens;
              outputTokens = data.usage.completion_tokens || outputTokens;
            }
          } catch (e) {
            // Skip unparseable chunks
          }
        }
      });

      azureRes.on('error', (err) => {
        console.error('[PROXY] Azure stream error:', err);
        res.end();
        reject(err);
      });

      azureRes.on('end', () => {
        if (!res.writableEnded) {
          res.end();
        }
        resolve();
      });
    });

    azureReq.on('error', (err) => {
      console.error('[PROXY] Request error:', err);
      res.end();
      reject(err);
    });

    azureReq.write(JSON.stringify(openaiReq));
    azureReq.end();
  });
}

async function handleNonStreaming(
  res: http.ServerResponse,
  options: https.RequestOptions,
  openaiReq: any,
  anthropicReq: any,
  verbose: boolean
): Promise<void> {
  return new Promise((resolve, reject) => {
    const azureReq = https.request(options, (azureRes) => {
      let body = '';

      azureRes.on('data', (chunk: Buffer) => {
        body += chunk.toString();
      });

      azureRes.on('end', () => {
        try {
          const data = JSON.parse(body);
          const choice = data.choices?.[0];

          if (!choice) {
            throw new Error('No response from Azure');
          }

          const content: any[] = [];

          // Add text content
          if (choice.message?.content) {
            content.push({ type: 'text', text: choice.message.content });
          }

          // Add tool calls
          if (choice.message?.tool_calls) {
            for (const tc of choice.message.tool_calls) {
              let input = {};
              try {
                input = JSON.parse(tc.function.arguments || '{}');
              } catch {
                input = {};
              }
              content.push({
                type: 'tool_use',
                id: tc.id,
                name: tc.function.name,
                input,
              });
            }
          }

          const response = {
            id: `msg_${Date.now()}`,
            type: 'message',
            role: 'assistant',
            content,
            model: anthropicReq.model,
            stop_reason: choice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn',
            stop_sequence: null,
            usage: {
              input_tokens: data.usage?.prompt_tokens || 0,
              output_tokens: data.usage?.completion_tokens || 0,
            },
          };

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(response));
          resolve();
        } catch (error: any) {
          console.error('[PROXY] Parse error:', error.message, body);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: error.message }));
          reject(error);
        }
      });
    });

    azureReq.on('error', (err) => {
      console.error('[PROXY] Request error:', err);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
      reject(err);
    });

    azureReq.write(JSON.stringify(openaiReq));
    azureReq.end();
  });
}

export function startProxy(config: ProxyConfig): Promise<void> {
  return new Promise((resolve) => {
    const server = createProxy(config);
    server.listen(config.port, '127.0.0.1', () => {
      resolve();
    });
  });
}
