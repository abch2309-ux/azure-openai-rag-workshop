import { DefaultAzureCredential, getBearerTokenProvider } from '@azure/identity';
import fp from 'fastify-plugin';
import { type BaseChatModel } from '@langchain/core/language_models/chat_models';
import { type VectorStore } from '@langchain/core/vectorstores';
import { type AIChatMessage, type AIChatCompletion } from '@microsoft/ai-chat-protocol';
import { MessageBuilder } from '../lib/message-builder.js';
import { type AppConfig } from './config.js';
import { AzureChatOpenAI, AzureOpenAIEmbeddings } from '@langchain/openai';
import { AzureAISearchVectorStore } from '@langchain/community/vectorstores/azure_aisearch';

const SYSTEM_MESSAGE_PROMPT = `Assistant helps the Consto Real Estate company customers with support questions regarding terms of service, privacy policy, and questions about support requests. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
Each source has a name followed by a colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example: [info1.txt]. Don't combine sources, list each source separately, for example: [info1.txt][info2.pdf].
`;

export class ChatService {
  tokenLimit: number = 4000;

  constructor(
    private config: AppConfig,
    private model: BaseChatModel,
    private vectorStore: VectorStore,
  ) {}

  async run(messages: AIChatMessage[]): Promise<AIChatCompletion> {
    // Get the content of the last message (the question)
    const query = messages[messages.length - 1].content;
    
    // Perform a vector similarity search
    const documents = await this.vectorStore.similaritySearch(query, 3);
    const results: string[] = [];

    // Process the search results
    for (const document of documents) {
      const source = document.metadata.source;
      const content = document.pageContent.replaceAll(/[\n\r]+/g, ' ');
      results.push(`${source}: ${content}`);
    }

    const content = results.join('\n');

    // Set the context with the system message
    const systemMessage = SYSTEM_MESSAGE_PROMPT;

    // Get the latest user message (the question), and inject the sources into it
    const userMessage = `${messages[messages.length - 1].content}\n\nSources:\n${content}`;

    // Create the messages prompt
    const messageBuilder = new MessageBuilder(systemMessage, this.config.azureOpenAiApiModelName);
    messageBuilder.appendMessage('user', userMessage);

    // Add the previous messages to the prompt, as long as we don't exceed the token limit
    for (const historyMessage of messages.slice(0, -1).reverse()) {
      if (messageBuilder.tokens > this.tokenLimit) {
        messageBuilder.popMessage();
        break;
      }
      messageBuilder.appendMessage(historyMessage.role, historyMessage.content);
    }

    // Debugging details
    const conversation = messageBuilder.messages.map((m) => `${m.role}: ${m.content}`).join('\n\n');
    const thoughts = `Search query:\n${query}\n\nConversation:\n${conversation}`.replaceAll('\n', '<br>');

    // Get the model completion
    const completion = await this.model.invoke(messageBuilder.getMessages());

    // Return the response in the Chat specification format
    return {
      message: {
        content: completion.content as string,
        role: 'assistant',
      },
      context: {
        data_points: results,
        thoughts: thoughts,
      },
    };
  }
}

export default fp(
  async (fastify, options) => {
    const config = fastify.config;

    // Use the current user identity to authenticate
    const credentials = new DefaultAzureCredential();

    // Set up OpenAI token provider
    const getToken = getBearerTokenProvider(credentials, 'https://cognitiveservices.azure.com/.default');
    const azureADTokenProvider = async () => {
      try {
        return await getToken();
      } catch {
        fastify.log.warn('Failed to get Azure OpenAI token, using dummy key');
        return '__dummy';
      }
    };

    // Set up LangChain.js clients
    fastify.log.info(`Using OpenAI at ${config.azureOpenAiApiEndpoint}`);

    const model = new AzureChatOpenAI({
      azureADTokenProvider,
      azureOpenAIBasePath: `${config.azureOpenAiApiEndpoint}/openai/deployments`,
      temperature: 0.7,
      maxTokens: 1024,
      n: 1,
    });
    const embeddings = new AzureOpenAIEmbeddings({
      azureADTokenProvider,
      azureOpenAIBasePath: `${config.azureOpenAiApiEndpoint}/openai/deployments`,
    });
    const vectorStore = new AzureAISearchVectorStore(embeddings, { credentials });

    // Initialize the ChatService instance
    const chatService = new ChatService(config, model, vectorStore);

    fastify.decorate('chat', chatService);
  },
  {
    name: 'chat',
    dependencies: ['config'],
  },
);

// When using .decorate you have to specify added properties for Typescript
declare module 'fastify' {
  export interface FastifyInstance {
    chat: ChatService;
  }
}
